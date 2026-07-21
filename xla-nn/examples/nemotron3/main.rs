// Inference example for the NVIDIA Nemotron 3 Nano 4B model,
// https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16.
//
// Nemotron-H style hybrid: each of the 42 layers holds a single mixer, either
// a Mamba2 SSM (21 layers), a plain MLP with a squared-relu activation (17
// layers), or full attention with GQA (4 layers), following a fixed pattern.
// The attention layers use no positional embedding at all: positional
// information only comes from the recurrent Mamba2 layers.
//
// The structure mirrors the qwen35 example: a prefill computation processes
// the whole padded context (with an XLA while loop running the Mamba2
// recurrence sequentially) and returns the first token plus the per-layer
// states, and a decode computation processes a single position. The state
// consists of the k/v caches for the attention layers and of the SSM state
// (kept in f32) plus the last conv1d inputs for the Mamba2 layers. All state
// tensors stay on the device across the generation steps.
use anyhow::{anyhow, Result};
use clap::Parser;

extern crate xla;
use xla::{ElementType, PjRtClient, PrimitiveType, Shape, XlaBuilder, XlaComputation, XlaOp};

use xla_nn::VarBuilder;

// Parameters 0 and 1 are reserved for the token ids and the last position.
const NUM_NON_WEIGHT_ARGS: usize = 2;

// Fixed context size the computations get compiled for, also the kv-cache
// length.
const CONTEXT_SIZE: usize = 128;
const T: i64 = CONTEXT_SIZE as i64;

const REPO: &str = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16";
// M: Mamba2, -: MLP, *: attention.
const PATTERN: &str = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-";
const HIDDEN_SIZE: i64 = 3136;
const VOCAB_SIZE: i64 = 131072;
const RMS_NORM_EPS: f32 = 1e-5;
// Attention.
const NUM_HEADS: i64 = 40;
const NUM_KV_HEADS: i64 = 8;
const HEAD_DIM: i64 = 128;
// MLP.
const INTERMEDIATE_SIZE: i64 = 12544;
// Mamba2.
const M_HEADS: i64 = 96;
const M_HEAD_DIM: i64 = 80;
const M_INTERMEDIATE: i64 = M_HEADS * M_HEAD_DIM; // 7680
const SSM_STATE: i64 = 128;
const N_GROUPS: i64 = 8;
const CONV_KERNEL_SIZE: i64 = 4;
const CONV_DIM: i64 = M_INTERMEDIATE + 2 * N_GROUPS * SSM_STATE; // 9728

fn linear(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    // x: [..., in], w: [out, in] -> [..., out]
    let x_rank = x.rank()? as i64;
    Ok(x.dot_general(w, &[x_rank - 1], &[1], &[], &[])?)
}

// Nemotron rms-norm: computed in f32, scaled by the plain weight, and cast
// back to the input dtype.
fn rms_norm(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let mean2 = (&x * &x)?.reduce_mean(&[-1], true)?;
    let x_norm = (&x * (mean2 + b.c0(RMS_NORM_EPS)?)?.rsqrt()?)?;
    let rank = x.rank()? as i64;
    let w =
        w.convert(PrimitiveType::F32)?.broadcast_in_dim(x.array_shape()?.dims(), &[rank - 1])?;
    Ok((x_norm * w)?.convert(dt)?)
}

// Softplus computed in f32 and rounded back to the input dtype, matching the
// single-op torch semantics on bf16 inputs.
fn softplus(x: &XlaOp) -> Result<XlaOp> {
    // Stable softplus: max(x, 0) + log1p(exp(-|x|))
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let zero = b.c0(0f32)?.broadcast(x.array_shape()?.dims())?;
    Ok((x.max(&zero)? + (x.abs()?.neg()?).exp()?.log1p()?)?.convert(dt)?)
}

// The MLP mixer: down(relu(up(x))^2).
struct Mlp {
    up_proj: XlaOp,
    down_proj: XlaOp,
}

impl Mlp {
    fn new(vb: &VarBuilder, p: &str) -> Result<Self> {
        let (h, i) = (HIDDEN_SIZE, INTERMEDIATE_SIZE);
        let up_proj = vb.var(&format!("{p}.up_proj.weight"), &[i, h])?;
        let down_proj = vb.var(&format!("{p}.down_proj.weight"), &[h, i])?;
        Ok(Self { up_proj, down_proj })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let b = x.builder();
        let up = linear(x, &self.up_proj)?;
        let zero = b.c0(0f32)?.convert(up.ty()?)?.broadcast(up.array_shape()?.dims())?;
        let relu = up.max(&zero)?;
        linear(&(&relu * &relu)?, &self.down_proj)
    }
}

// A single step of the Mamba2 (SSD) recurrence. s: [h, d, n], xdt_t: [h, d]
// (dt-scaled inputs), b_t/c_t: [h, n], da_t: [h] (the decay exp(dt_t * A)).
// Returns the updated state and the output [h, d]. The state is kept in f32,
// the xdt/b inputs and the output contraction use `dt` (bf16 in the decode
// step, f32 in the prefill scan) matching the reference implementation.
fn mamba_step(
    s: &XlaOp,
    xdt_t: &XlaOp,
    b_t: &XlaOp,
    c_t: &XlaOp,
    da_t: &XlaOp,
    dt: PrimitiveType,
) -> Result<(XlaOp, XlaOp)> {
    let (h, d, n) = (M_HEADS, M_HEAD_DIM, SSM_STATE);
    // s <- s * da_t + outer(xdt_t, b_t)
    let da_b = da_t.broadcast_in_dim(&[h, d, n], &[0])?;
    let dbx = (xdt_t.broadcast_in_dim(&[h, d, n], &[0, 1])?
        * b_t.broadcast_in_dim(&[h, d, n], &[0, 2])?)?;
    let s = ((s * da_b)? + dbx.convert(PrimitiveType::F32)?)?;
    // out_t = s . c_t, contracted over the state dim.
    let c_r = c_t.reshape(&[h, n, 1])?;
    let o_t = s.convert(dt)?.dot_general(&c_r, &[2], &[1], &[0], &[0])?.reshape(&[h, d])?;
    Ok((s, o_t))
}

// The Mamba2 mixer used on the "M" layers.
struct Mamba2 {
    in_proj: XlaOp,
    conv1d: XlaOp,
    conv1d_bias: XlaOp,
    a_log: XlaOp,
    d: XlaOp,
    dt_bias: XlaOp,
    norm: XlaOp,
    out_proj: XlaOp,
}

impl Mamba2 {
    fn new(vb: &VarBuilder, p: &str) -> Result<Self> {
        let proj_dim = M_INTERMEDIATE + CONV_DIM + M_HEADS;
        let in_proj = vb.var(&format!("{p}.in_proj.weight"), &[proj_dim, HIDDEN_SIZE])?;
        let conv1d = vb.var(&format!("{p}.conv1d.weight"), &[CONV_DIM, 1, CONV_KERNEL_SIZE])?;
        let conv1d_bias = vb.var(&format!("{p}.conv1d.bias"), &[CONV_DIM])?;
        let a_log = vb.var(&format!("{p}.A_log"), &[M_HEADS])?;
        let d = vb.var(&format!("{p}.D"), &[M_HEADS])?;
        let dt_bias = vb.var(&format!("{p}.dt_bias"), &[M_HEADS])?;
        let norm = vb.var(&format!("{p}.norm.weight"), &[M_INTERMEDIATE])?;
        let out_proj = vb.var(&format!("{p}.out_proj.weight"), &[HIDDEN_SIZE, M_INTERMEDIATE])?;
        Ok(Self { in_proj, conv1d, conv1d_bias, a_log, d, dt_bias, norm, out_proj })
    }

    // Depthwise causal conv1d with bias over the time dimension followed by a
    // silu activation. x: [t, CONV_DIM].
    fn causal_conv(&self, x: &XlaOp, t: i64) -> Result<XlaOp> {
        let b = x.builder();
        let w = self.conv1d.reshape(&[CONV_DIM, CONV_KERNEL_SIZE])?;
        let zero = b.c0(0f32)?.convert(x.ty()?)?;
        let mut out = None;
        for j in 0..CONV_KERNEL_SIZE {
            // shifted[i] = x[i - (k - 1 - j)]
            let shift = CONV_KERNEL_SIZE - 1 - j;
            let shifted = x.pad_in_dim(&zero, 0, shift, 0)?.slice_in_dim1(0, t, 0)?;
            let w_j = w.slice_in_dim1(j, j + 1, 1)?.reshape(&[CONV_DIM])?;
            let w_j = w_j.broadcast_in_dim(&[t, CONV_DIM], &[1])?;
            let contrib = (shifted * w_j)?;
            out = match out {
                None => Some(contrib),
                Some(acc) => Some((acc + contrib)?),
            };
        }
        let bias = self.conv1d_bias.broadcast_in_dim(&[t, CONV_DIM], &[1])?;
        Ok((out.unwrap() + bias)?.silu()?)
    }

    // Split the post-conv activations into x [t, h, d] and the B/C state
    // projections [t, h, n] with the per-group values repeated over the heads
    // of the group.
    fn split_xbc(&self, xbc: &XlaOp, t: i64, dt: PrimitiveType) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let x = xbc.slice_in_dim1(0, M_INTERMEDIATE, 1)?.reshape(&[t, M_HEADS, M_HEAD_DIM])?;
        let repeat = |start: i64| -> Result<XlaOp> {
            let heads_per_group = M_HEADS / N_GROUPS;
            let x = xbc.slice_in_dim1(start, start + N_GROUPS * SSM_STATE, 1)?;
            let x = x.reshape(&[t, N_GROUPS, SSM_STATE])?;
            Ok(x.broadcast_in_dim(&[t, N_GROUPS, heads_per_group, SSM_STATE], &[0, 1, 3])?
                .reshape(&[t, M_HEADS, SSM_STATE])?
                .convert(dt)?)
        };
        let b = repeat(M_INTERMEDIATE)?;
        let c = repeat(M_INTERMEDIATE + N_GROUPS * SSM_STATE)?;
        Ok((x.convert(dt)?, b, c))
    }

    // dt = softplus(dt_raw + dt_bias) rounded to bf16 as in the reference
    // implementation, and the decay exp(dt * A) in f32. Both [t, h] in f32.
    fn dt_and_decay(&self, dt_raw: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp)> {
        let dt_bias = self.dt_bias.broadcast_in_dim(&[t, M_HEADS], &[1])?;
        let dt = softplus(&(dt_raw + dt_bias)?)?.convert(PrimitiveType::F32)?;
        // A = -exp(A_log), da = exp(dt * A), in f32.
        let a = self.a_log.convert(PrimitiveType::F32)?.exp()?.neg()?;
        let a = a.broadcast_in_dim(&[t, M_HEADS], &[1])?;
        let da = (&dt * a)?.exp()?;
        Ok((dt, da))
    }

    // Gated group rms-norm followed by the output projection. y: [t, h, d]
    // (f32 from the prefill scan, model dtype from the decode step), gate:
    // [t, h*d] in the model dtype. The norm is computed in f32 over groups of
    // h*d/N_GROUPS channels: y * silu(gate) normalized per group, * weight.
    fn output(&self, y: &XlaOp, gate: &XlaOp, t: i64) -> Result<XlaOp> {
        let b = y.builder();
        let dt = gate.ty()?;
        let group_size = M_INTERMEDIATE / N_GROUPS;
        let y = y.reshape(&[t, M_INTERMEDIATE])?.convert(PrimitiveType::F32)?;
        let gated = (y * gate.convert(PrimitiveType::F32)?.silu()?)?;
        let grouped = gated.reshape(&[t, N_GROUPS, group_size])?;
        let mean2 = (&grouped * &grouped)?.reduce_mean(&[-1], true)?;
        let normed = (grouped * (mean2 + b.c0(RMS_NORM_EPS)?)?.rsqrt()?)?;
        let normed = normed.reshape(&[t, M_INTERMEDIATE])?;
        let w =
            self.norm.convert(PrimitiveType::F32)?.broadcast_in_dim(&[t, M_INTERMEDIATE], &[1])?;
        linear(&(normed * w)?.convert(dt)?, &self.out_proj)
    }

    // The SSD recurrence, computed sequentially over the first n positions
    // with an XLA while loop, everything in f32. xdt: [T, h, d] (dt-scaled
    // inputs), b/c: [T, h, n], da: [T, h]. Returns the per-step outputs
    // [T, h, d] (zeros beyond n) and the final state [h, d, n].
    fn ssm_scan(
        &self,
        builder: &XlaBuilder,
        xdt: &XlaOp,
        b: &XlaOp,
        c: &XlaOp,
        da: &XlaOp,
        n: &XlaOp,
    ) -> Result<(XlaOp, XlaOp)> {
        let (h, d, s_dim) = (M_HEADS, M_HEAD_DIM, SSM_STATE);
        let state_shape = Shape::tuple(vec![
            Shape::array_with_type(ElementType::S32, vec![]), // t
            Shape::array::<f32>(vec![h, d, s_dim]),           // ssm state
            Shape::array::<f32>(vec![T, h, d]),               // outputs
            Shape::array::<f32>(vec![T, h, d]),               // xdt
            Shape::array::<f32>(vec![T, h, s_dim]),           // b
            Shape::array::<f32>(vec![T, h, s_dim]),           // c
            Shape::array::<f32>(vec![T, h]),                  // da
            Shape::array_with_type(ElementType::S32, vec![]), // n
        ]);
        let cond = {
            let b = XlaBuilder::new("cond");
            let state = b.parameter_s(0, &state_shape, "state")?;
            state.get_tuple_element(0)?.lt(&state.get_tuple_element(7)?)?.build()?
        };
        let body = {
            let bb = XlaBuilder::new("body");
            let state = bb.parameter_s(0, &state_shape, "state")?;
            let t = state.get_tuple_element(0)?;
            let s = state.get_tuple_element(1)?;
            let out = state.get_tuple_element(2)?;
            let xdt = state.get_tuple_element(3)?;
            let b = state.get_tuple_element(4)?;
            let c = state.get_tuple_element(5)?;
            let da = state.get_tuple_element(6)?;
            let n = state.get_tuple_element(7)?;
            let zero = bb.c0(0i32)?;
            let at_t = |x: &XlaOp, d2: i64| -> Result<XlaOp> {
                Ok(x.dynamic_slice(&[&t, &zero, &zero], &[1, h, d2])?.reshape(&[h, d2])?)
            };
            let xdt_t = at_t(&xdt, d)?;
            let b_t = at_t(&b, s_dim)?;
            let c_t = at_t(&c, s_dim)?;
            let da_t = da.dynamic_slice(&[&t, &zero], &[1, h])?.reshape(&[h])?;
            let (s, o_t) = mamba_step(&s, &xdt_t, &b_t, &c_t, &da_t, PrimitiveType::F32)?;
            let out = out.dynamic_update_slice(&o_t.reshape(&[1, h, d])?, &[&t, &zero, &zero])?;
            let t = (t + bb.c0(1i32)?)?;
            bb.tuple(&[t, s, out, xdt, b, c, da, n])?.build()?
        };
        let init_s = builder.c0(0f32)?.broadcast(&[h, d, s_dim])?;
        let init_out = builder.c0(0f32)?.broadcast(&[T, h, d])?;
        let init = builder.tuple(&[
            builder.c0(0i32)?,
            init_s,
            init_out,
            xdt.clone(),
            b.clone(),
            c.clone(),
            da.clone(),
            n.clone(),
        ])?;
        let result = XlaOp::while_(cond, body, init)?;
        Ok((result.get_tuple_element(2)?, result.get_tuple_element(1)?))
    }

    // Full-context forward. x: [T, HIDDEN]. Returns the outputs [T, HIDDEN],
    // the SSM state after processing position last_pos, and the conv1d inputs
    // at positions last_pos-2..last_pos (zero padded on the left).
    fn forward_prefill(
        &self,
        builder: &XlaBuilder,
        x: &XlaOp,
        last_pos: &XlaOp,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let b = x.builder();
        let f32t = PrimitiveType::F32;
        let proj = linear(x, &self.in_proj)?;
        let gate = proj.slice_in_dim1(0, M_INTERMEDIATE, 1)?;
        let pre_conv = proj.slice_in_dim1(M_INTERMEDIATE, M_INTERMEDIATE + CONV_DIM, 1)?;
        let dt_raw =
            proj.slice_in_dim1(M_INTERMEDIATE + CONV_DIM, M_INTERMEDIATE + CONV_DIM + M_HEADS, 1)?;
        let xbc = self.causal_conv(&pre_conv, T)?;
        // The whole recurrence runs in f32 as in the reference implementation.
        let (xh, bs, cs) = self.split_xbc(&xbc, T, f32t)?;
        let (dt, da) = self.dt_and_decay(&dt_raw, T)?;
        let xdt = (&xh
            * dt.reshape(&[T, M_HEADS, 1])?
                .broadcast_in_dim(&[T, M_HEADS, M_HEAD_DIM], &[0, 1, 2])?)?;
        let n = (last_pos + b.c0(1i32)?)?;
        let (out, s) = self.ssm_scan(builder, &xdt, &bs, &cs, &da, &n)?;
        // The D skip connection uses the un-scaled inputs.
        let d_skip = self.d.convert(f32t)?.broadcast_in_dim(&[T, M_HEADS, M_HEAD_DIM], &[1])?;
        let y = (out + (xh * d_skip)?)?;
        let y = self.output(&y, &gate, T)?;
        // The conv state holds the last CONV_KERNEL_SIZE-1 conv inputs; the
        // zero padding covers prompts shorter than the kernel and keeps the
        // dynamic slice in bounds.
        let zero_f = b.c0(0f32)?.convert(pre_conv.ty()?)?;
        let zero_i = b.c0(0i32)?;
        let padded = pre_conv.pad_in_dim(&zero_f, 0, CONV_KERNEL_SIZE - 1, 0)?;
        let conv_state = padded.dynamic_slice(&[&n, &zero_i], &[CONV_KERNEL_SIZE - 1, CONV_DIM])?;
        Ok((y, s, conv_state))
    }

    // Single position forward. x: [1, HIDDEN], s: [h, d, n],
    // conv_state: [CONV_KERNEL_SIZE-1, CONV_DIM]. Returns the output and the
    // updated states. The xdt/b/c values stay in bf16 here (with an f32 state
    // and decay), matching the reference decode path.
    fn forward_step(
        &self,
        x: &XlaOp,
        s: &XlaOp,
        conv_state: &XlaOp,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let dt_model = x.ty()?;
        let proj = linear(x, &self.in_proj)?;
        let gate = proj.slice_in_dim1(0, M_INTERMEDIATE, 1)?;
        let pre_conv = proj.slice_in_dim1(M_INTERMEDIATE, M_INTERMEDIATE + CONV_DIM, 1)?;
        let dt_raw =
            proj.slice_in_dim1(M_INTERMEDIATE + CONV_DIM, M_INTERMEDIATE + CONV_DIM + M_HEADS, 1)?;
        let window = conv_state.concat_in_dim(&[&pre_conv], 0)?;
        let new_conv_state = window.slice_in_dim1(1, CONV_KERNEL_SIZE, 0)?;
        // out[c] = sum_j window[j, c] * w[c, j]
        let w = self.conv1d.reshape(&[CONV_DIM, CONV_KERNEL_SIZE])?.swap_dims(0, 1)?;
        let bias = self.conv1d_bias.reshape(&[1, CONV_DIM])?;
        let xbc = (((window * w)?.reduce_sum(&[0], true)? + bias)?).silu()?;
        let (xh, bs, cs) = self.split_xbc(&xbc, 1, dt_model)?;
        let (dt, da) = self.dt_and_decay(&dt_raw, 1)?;
        // xdt is computed in the model dtype in the decode path.
        let dt_b = dt
            .convert(dt_model)?
            .reshape(&[1, M_HEADS, 1])?
            .broadcast_in_dim(&[1, M_HEADS, M_HEAD_DIM], &[0, 1, 2])?;
        let xdt = (&xh * dt_b)?;
        let (s, o_t) = mamba_step(
            s,
            &xdt.reshape(&[M_HEADS, M_HEAD_DIM])?,
            &bs.reshape(&[M_HEADS, SSM_STATE])?,
            &cs.reshape(&[M_HEADS, SSM_STATE])?,
            &da.reshape(&[M_HEADS])?,
            dt_model,
        )?;
        let d_skip = self.d.broadcast_in_dim(&[M_HEADS, M_HEAD_DIM], &[0])?;
        let y = (o_t + (xh.reshape(&[M_HEADS, M_HEAD_DIM])? * d_skip)?)?;
        let y = self.output(&y.reshape(&[1, M_HEADS, M_HEAD_DIM])?, &gate, 1)?;
        Ok((y, s, new_conv_state))
    }
}

// Full attention with GQA and no positional embedding.
struct Attention {
    q_proj: XlaOp,
    k_proj: XlaOp,
    v_proj: XlaOp,
    o_proj: XlaOp,
}

impl Attention {
    fn new(vb: &VarBuilder, p: &str) -> Result<Self> {
        let h = HIDDEN_SIZE;
        let q_proj = vb.var(&format!("{p}.q_proj.weight"), &[NUM_HEADS * HEAD_DIM, h])?;
        let k_proj = vb.var(&format!("{p}.k_proj.weight"), &[NUM_KV_HEADS * HEAD_DIM, h])?;
        let v_proj = vb.var(&format!("{p}.v_proj.weight"), &[NUM_KV_HEADS * HEAD_DIM, h])?;
        let o_proj = vb.var(&format!("{p}.o_proj.weight"), &[h, NUM_HEADS * HEAD_DIM])?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj })
    }

    fn qkv(&self, x: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let q = linear(x, &self.q_proj)?.reshape(&[t, NUM_HEADS, HEAD_DIM])?;
        let k = linear(x, &self.k_proj)?.reshape(&[t, NUM_KV_HEADS, HEAD_DIM])?;
        let v = linear(x, &self.v_proj)?.reshape(&[t, NUM_KV_HEADS, HEAD_DIM])?;
        Ok((q, k, v))
    }

    // Attend from q [t, h, D] over k/v [s, kvh, D] with mask [t, s]. The
    // softmax is computed in f32 as in the reference implementation.
    fn attend(
        &self,
        q: &XlaOp,
        k: &XlaOp,
        v: &XlaOp,
        mask: &XlaOp,
        t: i64,
        s: i64,
    ) -> Result<XlaOp> {
        let b = q.builder();
        let (nh, nkv) = (NUM_HEADS, NUM_KV_HEADS);
        // [t, h, d] -> [h, t, d], and repeat the kv heads for GQA.
        let q = q.swap_dims(0, 1)?;
        let repeat_kv = |x: &XlaOp| -> Result<XlaOp> {
            let x = x.swap_dims(0, 1)?;
            let groups = nh / nkv;
            Ok(x.broadcast_in_dim(&[nkv, groups, s, HEAD_DIM], &[0, 2, 3])?
                .reshape(&[nh, s, HEAD_DIM])?)
        };
        let k = repeat_kv(k)?;
        let v = repeat_kv(v)?;
        let dt = q.ty()?;
        let scale = b.c0(1f32 / (HEAD_DIM as f32).sqrt())?.convert(dt)?;
        let scores = (q.dot_general(&k, &[2], &[2], &[0], &[0])? * scale)?;
        let mask_b = mask.convert(dt)?.broadcast_in_dim(&[nh, t, s], &[1, 2])?;
        let probs = (scores + mask_b)?.convert(PrimitiveType::F32)?.softmax(-1)?.convert(dt)?;
        let ctx = probs.dot_general(&v, &[2], &[1], &[0], &[0])?;
        let ctx = ctx.swap_dims(0, 1)?.reshape(&[t, nh * HEAD_DIM])?;
        linear(&ctx, &self.o_proj)
    }

    // Full-context forward. Returns the outputs [T, HIDDEN] and the k/v
    // caches [T, kvh, D] (positions beyond the prompt hold garbage and get
    // overwritten by the decode steps).
    fn forward_prefill(&self, x: &XlaOp, mask: &XlaOp) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let (q, k, v) = self.qkv(x, T)?;
        let y = self.attend(&q, &k, &v, mask, T, T)?;
        Ok((y, k, v))
    }

    // Single position forward at position pos, mask: [1, T] allowing
    // positions <= pos. Returns the output and the updated k/v caches.
    fn forward_step(
        &self,
        x: &XlaOp,
        pos: &XlaOp,
        mask: &XlaOp,
        k_cache: &XlaOp,
        v_cache: &XlaOp,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let b = x.builder();
        let (q, k, v) = self.qkv(x, 1)?;
        let zero = b.c0(0i32)?;
        let k_cache = k_cache.dynamic_update_slice(&k, &[pos, &zero, &zero])?;
        let v_cache = v_cache.dynamic_update_slice(&v, &[pos, &zero, &zero])?;
        let y = self.attend(&q, &k_cache, &v_cache, mask, 1, T)?;
        Ok((y, k_cache, v_cache))
    }
}

enum Mixer {
    Mamba(Mamba2),
    Attn(Attention),
    Mlp(Mlp),
}

struct Block {
    norm: XlaOp,
    mixer: Mixer,
}

struct Model {
    embed: XlaOp,
    layers: Vec<Block>,
    final_ln: XlaOp,
    lm_head: XlaOp,
}

impl Model {
    // Weight declaration order must be identical between the prefill and the
    // decode builders as they share a single buffer list.
    fn new(vb: &VarBuilder) -> Result<Self> {
        let embed = vb.var("backbone.embeddings.weight", &[VOCAB_SIZE, HIDDEN_SIZE])?;
        let mut layers = Vec::with_capacity(PATTERN.len());
        for (layer_idx, kind) in PATTERN.chars().enumerate() {
            let p = format!("backbone.layers.{layer_idx}");
            let norm = vb.var(&format!("{p}.norm.weight"), &[HIDDEN_SIZE])?;
            let mixer = match kind {
                'M' => Mixer::Mamba(Mamba2::new(vb, &format!("{p}.mixer"))?),
                '*' => Mixer::Attn(Attention::new(vb, &format!("{p}.mixer"))?),
                '-' => Mixer::Mlp(Mlp::new(vb, &format!("{p}.mixer"))?),
                _ => anyhow::bail!("invalid layer pattern char {kind}"),
            };
            layers.push(Block { norm, mixer });
        }
        let final_ln = vb.var("backbone.norm_f.weight", &[HIDDEN_SIZE])?;
        let lm_head = vb.var("lm_head.weight", &[VOCAB_SIZE, HIDDEN_SIZE])?;
        Ok(Self { embed, layers, final_ln, lm_head })
    }
}

// Causal mask as a constant, [T, T]. Row i allows positions j <= i.
fn causal_mask(builder: &XlaBuilder) -> Result<XlaOp> {
    let mut mask_data = vec![0f32; (T * T) as usize];
    for i in 0..T as usize {
        for j in 0..T as usize {
            if j > i {
                mask_data[i * T as usize + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(builder.c1(&mask_data)?.reshape(&[T, T])?)
}

// The prefill computation: full padded context in, next token plus all the
// per-layer states out.
fn build_prefill(builder: &XlaBuilder, vb: &VarBuilder) -> Result<XlaComputation> {
    let tokens = builder.parameter(0, ElementType::S32, &[T], "tokens")?;
    let last_pos = builder.parameter(1, ElementType::S32, &[], "last_pos")?;
    let model = Model::new(vb)?;
    let mask = causal_mask(builder)?;

    let mut x = model.embed.take(&tokens, 0)?;
    let mut states = Vec::new();
    for layer in model.layers.iter() {
        let x_norm = rms_norm(&x, &layer.norm)?;
        let mixed = match &layer.mixer {
            Mixer::Mamba(mamba) => {
                let (y, s, conv_state) = mamba.forward_prefill(builder, &x_norm, &last_pos)?;
                states.push(s);
                states.push(conv_state);
                y
            }
            Mixer::Attn(attn) => {
                let (y, k_cache, v_cache) = attn.forward_prefill(&x_norm, &mask)?;
                states.push(k_cache);
                states.push(v_cache);
                y
            }
            Mixer::Mlp(mlp) => mlp.forward(&x_norm)?,
        };
        x = (x + mixed)?;
    }

    let x = rms_norm(&x, &model.final_ln)?;
    // Only the logits for the last position are needed.
    let zero = builder.c0(0i32)?;
    let x_last = x.dynamic_slice(&[&last_pos, &zero], &[1, HIDDEN_SIZE])?;
    let logits = linear(&x_last, &model.lm_head)?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?;

    let mut outputs = vec![next_token];
    outputs.extend(states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// The decode computation: a single token at a given position plus the
// per-layer states in, next token plus the updated states out. The states are
// passed as parameters after the weights so that the weight parameter indices
// match the prefill computation.
fn build_decode(builder: &XlaBuilder, vb: &VarBuilder) -> Result<XlaComputation> {
    let token = builder.parameter(0, ElementType::S32, &[1], "token")?;
    let pos = builder.parameter(1, ElementType::S32, &[], "pos")?;
    let model = Model::new(vb)?;

    let mut param_idx = (NUM_NON_WEIGHT_ARGS + vb.num_vars()) as i64;
    let mut state_param = |name: String, ty: ElementType, dims: &[i64]| -> Result<XlaOp> {
        let op = builder.parameter(param_idx, ty, dims, &name)?;
        param_idx += 1;
        Ok(op)
    };
    let dtype = vb.dtype();
    let mut states = Vec::new();
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        match &layer.mixer {
            Mixer::Mamba(_) => {
                // The SSM state is kept in f32, the conv state holds raw conv
                // inputs and uses the model dtype.
                let s_dims = [M_HEADS, M_HEAD_DIM, SSM_STATE];
                states.push(state_param(
                    format!("layers.{layer_idx}.ssm_state"),
                    ElementType::F32,
                    &s_dims,
                )?);
                let c_dims = [CONV_KERNEL_SIZE - 1, CONV_DIM];
                states.push(state_param(format!("layers.{layer_idx}.conv_state"), dtype, &c_dims)?);
            }
            Mixer::Attn(_) => {
                let dims = [T, NUM_KV_HEADS, HEAD_DIM];
                states.push(state_param(format!("layers.{layer_idx}.k_cache"), dtype, &dims)?);
                states.push(state_param(format!("layers.{layer_idx}.v_cache"), dtype, &dims)?);
            }
            Mixer::Mlp(_) => {}
        }
    }

    let mask = causal_mask(builder)?;
    let zero = builder.c0(0i32)?;
    let mask = mask.dynamic_slice(&[&pos, &zero], &[1, T])?;

    let mut x = model.embed.take(&token, 0)?;
    let mut new_states = Vec::new();
    let mut state_idx = 0;
    for layer in model.layers.iter() {
        let x_norm = rms_norm(&x, &layer.norm)?;
        let mixed = match &layer.mixer {
            Mixer::Mamba(mamba) => {
                let (s, conv_state) = (&states[state_idx], &states[state_idx + 1]);
                state_idx += 2;
                let (y, s, conv_state) = mamba.forward_step(&x_norm, s, conv_state)?;
                new_states.push(s);
                new_states.push(conv_state);
                y
            }
            Mixer::Attn(attn) => {
                let (k_cache, v_cache) = (&states[state_idx], &states[state_idx + 1]);
                state_idx += 2;
                let (y, k_cache, v_cache) =
                    attn.forward_step(&x_norm, &pos, &mask, k_cache, v_cache)?;
                new_states.push(k_cache);
                new_states.push(v_cache);
                y
            }
            Mixer::Mlp(mlp) => mlp.forward(&x_norm)?,
        };
        x = (x + mixed)?;
    }

    let x = rms_norm(&x, &model.final_ln)?;
    let logits = linear(&x, &model.lm_head)?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?;

    let mut outputs = vec![next_token];
    outputs.extend(new_states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// Download the tokenizer and weights from the hugging face hub, using the
// local cache if they have already been fetched.
fn hub_model_files() -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let client = hf_hub::HFClientSync::new()?;
    let (owner, name) = REPO.split_once('/').ok_or_else(|| anyhow!("invalid repo"))?;
    let repo = client.model(owner, name);
    let tokenizer = repo
        .download_file()
        .filename("tokenizer.json")
        .progress(DownloadProgress::new("tokenizer.json"))
        .send()?;
    let weights = repo
        .download_file()
        .filename("model.safetensors")
        .progress(DownloadProgress::new("model.safetensors"))
        .send()?;
    Ok((tokenizer, vec![weights]))
}

#[derive(Parser, Debug)]
struct Args {
    /// Run on cpu rather than on gpu (still in bf16, mostly for debugging).
    #[arg(long)]
    cpu: bool,

    /// The prompt used for generation.
    #[arg(long, default_value = "What is the capital of France? Answer in one word.")]
    prompt: String,

    /// The maximum number of tokens to generate.
    #[arg(long, default_value_t = 30)]
    sample_len: usize,

    /// Feed the raw prompt to the model rather than using the chat template.
    #[arg(long)]
    raw_prompt: bool,

    /// Cache file for the gpu gemm autotuner results: loaded when the file
    /// exists, written otherwise. See the qwen35 example for details.
    #[arg(long)]
    autotune_cache: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (at_load, at_dump) = match &args.autotune_cache {
        None => (None, None),
        Some(path) => {
            let path = path.to_str().ok_or_else(|| anyhow!("non-utf8 autotune-cache path"))?;
            if std::path::Path::new(path).exists() {
                (Some(path.to_string()), None)
            } else {
                (None, Some(path.to_string()))
            }
        }
    };
    xla::set_tf_min_log_level(xla::TfLogLevel::Warning);
    xla::set_min_log_level(xla::TfLogLevel::Warning);
    let client = PjRtClient::auto(args.cpu)?;
    println!(
        "platform: {} {}, model: {REPO}, dtype: bf16",
        client.platform_name(),
        client.platform_version(),
    );

    let (tokenizer_path, weights_paths) = hub_model_files()?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("cannot load tokenizer: {e}"))?;
    // The chat template with thinking disabled.
    let prompt = if args.raw_prompt {
        args.prompt.clone()
    } else {
        format!(
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think></think>",
            args.prompt
        )
    };
    let encoded = tokenizer.encode(prompt, false).map_err(|e| anyhow!("tokenizer error: {e}"))?;
    let mut tokens: Vec<i32> = encoded.get_ids().iter().map(|&t| t as i32).collect();
    println!("prompt has {} tokens", tokens.len());
    if tokens.is_empty() || tokens.len() >= CONTEXT_SIZE {
        anyhow::bail!("prompt length must be in [1, {}]", CONTEXT_SIZE - 1)
    }
    // <|im_end|> (the tokenizer eos) and </s> (the generation config eos).
    let stop_tokens: Vec<i32> = vec![11, 2];

    let start = std::time::Instant::now();
    let prefill_builder = XlaBuilder::new("nemotron3-prefill");
    let vb = VarBuilder::new(&prefill_builder, ElementType::Bf16, NUM_NON_WEIGHT_ARGS);
    let prefill = build_prefill(&prefill_builder, &vb)?;
    let decode_builder = XlaBuilder::new("nemotron3-decode");
    let decode_vb = VarBuilder::new(&decode_builder, ElementType::Bf16, NUM_NON_WEIGHT_ARGS);
    let decode = build_decode(&decode_builder, &decode_vb)?;
    println!("built the computations in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let (at_load, at_dump) = (at_load.as_deref(), at_dump.as_deref());
    let prefill_exe = client.compile_with_autotune_cache(&prefill, at_load, at_dump)?;
    let decode_exe = client.compile_with_autotune_cache(&decode, at_load, at_dump)?;
    println!("compiled the executables in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let weight_buffers = vb.load_buffers(&weights_paths, &client)?;
    println!("loaded {} weights in {:?}", weight_buffers.len(), start.elapsed());

    let start = std::time::Instant::now();
    // Prefill: process the whole prompt, get the first token and the states.
    let mut padded = tokens.clone();
    padded.resize(CONTEXT_SIZE, 0);
    let token_buffer = client.buffer_from_host_buffer(&padded, &[CONTEXT_SIZE], None)?;
    let pos_buffer = client.buffer_from_host_buffer(&[tokens.len() as i32 - 1], &[], None)?;
    let mut inputs: Vec<&xla::PjRtBuffer> = vec![&token_buffer, &pos_buffer];
    inputs.extend(weight_buffers.iter());
    let prefill_outputs = prefill_exe
        .execute_b(&inputs)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no execution result"))?;
    prefill_outputs[0].to_literal_sync()?;
    println!("prefill ({} tokens) in {:?}", tokens.len(), start.elapsed());
    let start = std::time::Instant::now();

    // Decode: one token at a time, the state buffers stay on the device, and
    // the generated token is chained on the device as in the qwen35 example.
    let prompt_len = tokens.len();
    let mut generated = 0usize;
    let mut in_flight = prefill_outputs;
    loop {
        // in_flight produces the token at position prompt_len + generated.
        let pos = prompt_len + generated;
        let next_outputs = if generated + 1 < args.sample_len && pos + 1 < CONTEXT_SIZE {
            let pos_buffer = client.buffer_from_host_buffer(&[pos as i32], &[], None)?;
            let mut inputs: Vec<&xla::PjRtBuffer> = vec![&in_flight[0], &pos_buffer];
            inputs.extend(weight_buffers.iter());
            inputs.extend(in_flight[1..].iter());
            Some(
                decode_exe
                    .execute_b(&inputs)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow!("no execution result"))?,
            )
        } else {
            None
        };
        let next_token = in_flight[0].to_literal_sync()?.to_vec::<i32>()?[0];
        tokens.push(next_token);
        generated += 1;
        match next_outputs {
            Some(o) if !stop_tokens.contains(&next_token) => in_flight = o,
            _ => break,
        }
    }
    let decode_steps = generated - 1;
    let dt = start.elapsed();
    if decode_steps > 0 {
        let tok_s = decode_steps as f64 / dt.as_secs_f64();
        println!("decoded {decode_steps} tokens in {dt:?} -> {tok_s:.1} tok/s");
    }
    println!("generated ids: {:?}", &tokens[tokens.len() - generated..]);

    let all_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&all_ids, false).map_err(|e| anyhow!("tokenizer error: {e}"))?;
    println!("----\n{text}\n----");
    Ok(())
}

fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1} kB", b / 1e3)
    } else {
        format!("{b} B")
    }
}

struct DownloadProgress {
    label: String,
    total: std::sync::atomic::AtomicU64,
    done: std::sync::atomic::AtomicU64,
}

impl DownloadProgress {
    fn new(label: impl Into<String>) -> Self {
        use std::sync::atomic::AtomicU64;
        Self { label: label.into(), total: AtomicU64::new(0), done: AtomicU64::new(0) }
    }

    fn render(&self) {
        use std::io::Write;
        use std::sync::atomic::Ordering;
        let done = self.done.load(Ordering::Relaxed);
        let total = self.total.load(Ordering::Relaxed);
        if total > 0 {
            let pct = 100.0 * done as f64 / total as f64;
            eprint!(
                "\r  {}: {pct:5.1}% ({} / {})    ",
                self.label,
                human_bytes(done),
                human_bytes(total)
            );
        } else {
            eprint!("\r  {}: {}    ", self.label, human_bytes(done));
        }
        let _ = std::io::stderr().flush();
    }
}

impl hf_hub::progress::ProgressHandler for DownloadProgress {
    fn on_progress(&self, event: &hf_hub::progress::ProgressEvent) {
        use hf_hub::progress::{DownloadEvent, ProgressEvent};
        use std::sync::atomic::Ordering;
        let ProgressEvent::Download(event) = event else { return };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.total.store(*total_bytes, Ordering::Relaxed);
                self.render();
            }
            DownloadEvent::Progress { files } => {
                if let Some(f) = files.iter().max_by_key(|f| f.bytes_completed) {
                    if f.total_bytes > 0 {
                        self.total.store(f.total_bytes, Ordering::Relaxed);
                    }
                    self.done.store(f.bytes_completed, Ordering::Relaxed);
                    self.render();
                }
            }
            DownloadEvent::AggregateProgress { bytes_completed, total_bytes, .. } => {
                self.total.store(*total_bytes, Ordering::Relaxed);
                self.done.store(*bytes_completed, Ordering::Relaxed);
                self.render();
            }
            DownloadEvent::Complete => {
                let total = self.total.load(Ordering::Relaxed);
                self.done.store(total, Ordering::Relaxed);
                self.render();
                eprintln!();
            }
        }
    }
}
