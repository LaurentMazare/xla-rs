// Inference example for the Qwen3.5-2B model, https://huggingface.co/Qwen/Qwen3.5-2B
//
// Qwen3.5 is a hybrid architecture: three out of four decoder layers use a gated
// DeltaNet linear attention (implemented below with a XLA while loop over the
// sequence), the remaining ones use full attention with GQA, a partial rotary
// embedding, and a sigmoid output gate.
//
// The model files are downloaded automatically from the hugging face hub, the
// example requires the `hf-hub` feature (--features hf-hub).
//
// Two computations are compiled: a prefill one that processes the whole
// (padded) prompt and returns the per-layer state together with the first
// generated token, and a decode one that processes a single position using and
// updating that state. The state consists of the k/v caches (post-rope) for
// the full attention layers, and of the recurrent state plus the last conv1d
// inputs for the DeltaNet layers. All state tensors stay on the device across
// the generation steps.
use anyhow::{anyhow, Result};
use clap::Parser;

extern crate xla;
use xla::{ElementType, PjRtClient, PrimitiveType, Shape, XlaBuilder, XlaComputation, XlaOp};

mod var_store;
use var_store::{VarBuilder, NUM_NON_WEIGHT_ARGS};

// Fixed context size the computations get compiled for, also the kv-cache
// length.
const CONTEXT_SIZE: usize = 128;

// Configuration for Qwen3.5-2B (text model part).
const HIDDEN_SIZE: i64 = 2048;
const INTERMEDIATE_SIZE: i64 = 6144;
const NUM_LAYERS: usize = 24;
const FULL_ATTENTION_INTERVAL: usize = 4;
const VOCAB_SIZE: i64 = 248320;
const RMS_NORM_EPS: f32 = 1e-6;
// Full attention.
const NUM_HEADS: i64 = 8;
const NUM_KV_HEADS: i64 = 2;
const HEAD_DIM: i64 = 256;
const ROPE_THETA: f64 = 1e7;
const ROTARY_DIM: i64 = 64; // partial_rotary_factor 0.25
                            // Linear attention (gated DeltaNet).
const LIN_NUM_HEADS: i64 = 16;
const LIN_KEY_DIM: i64 = 128;
const LIN_VALUE_DIM: i64 = 128;
const CONV_KERNEL_SIZE: i64 = 4;
const CONV_DIM: i64 = 2 * LIN_NUM_HEADS * LIN_KEY_DIM + LIN_NUM_HEADS * LIN_VALUE_DIM;

const T: i64 = CONTEXT_SIZE as i64;

fn linear(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    // x: [..., in], w: [out, in] -> [..., out]
    let x_rank = x.rank()? as i64;
    Ok(x.dot_general(w, &[x_rank - 1], &[1], &[], &[])?)
}

// Qwen3.5 rms-norm uses a zero-centered weight, i.e. scales by (1 + weight).
// As in the reference implementation, the norm is computed in f32 and the
// result cast back to the input dtype.
fn rms_norm(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let mean2 = (&x * &x)?.reduce_mean(&[-1], true)?;
    let x_norm = (&x * (mean2 + b.c0(RMS_NORM_EPS)?)?.rsqrt()?)?;
    let rank = x.rank()? as i64;
    let w =
        w.convert(PrimitiveType::F32)?.broadcast_in_dim(x.array_shape()?.dims(), &[rank - 1])?;
    Ok((x_norm * (w + b.c0(1f32)?)?)?.convert(dt)?)
}

fn softplus(x: &XlaOp) -> Result<XlaOp> {
    // Stable softplus: max(x, 0) + log1p(exp(-|x|))
    let b = x.builder();
    let zero = b.c0(0f32)?.broadcast(x.array_shape()?.dims())?;
    Ok((x.max(&zero)? + (x.abs()?.neg()?).exp()?.log1p()?)?)
}

fn l2_norm(x: &XlaOp) -> Result<XlaOp> {
    // Note: this normalizes by the sum of squares, not the mean.
    let b = x.builder();
    let sum2 = (x * x)?.reduce_sum(&[-1], true)?;
    Ok((x * (sum2 + b.c0(1e-6f32)?)?.rsqrt()?)?)
}

struct Mlp {
    gate_proj: XlaOp,
    up_proj: XlaOp,
    down_proj: XlaOp,
}

impl Mlp {
    fn new(vb: &VarBuilder, p: &str) -> Result<Self> {
        let gate_proj =
            vb.var(&format!("{p}.gate_proj.weight"), &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?;
        let up_proj = vb.var(&format!("{p}.up_proj.weight"), &[INTERMEDIATE_SIZE, HIDDEN_SIZE])?;
        let down_proj =
            vb.var(&format!("{p}.down_proj.weight"), &[HIDDEN_SIZE, INTERMEDIATE_SIZE])?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let gated = (linear(x, &self.gate_proj)?.silu()? * linear(x, &self.up_proj)?)?;
        linear(&gated, &self.down_proj)
    }
}

// A single step of the gated delta rule. s: [h, dk, dv], q_t/k_t: [h, dk],
// v_t: [h, dv], exp_g_t/beta_t: [h]. Returns the updated state and the output.
fn delta_step(
    s: &XlaOp,
    q_t: &XlaOp,
    k_t: &XlaOp,
    v_t: &XlaOp,
    exp_g_t: &XlaOp,
    beta_t: &XlaOp,
) -> Result<(XlaOp, XlaOp)> {
    let h = LIN_NUM_HEADS;
    let (dk, dv) = (LIN_KEY_DIM, LIN_VALUE_DIM);
    let bcast_h = |x: &XlaOp| x.broadcast_in_dim(&[h, dk, dv], &[0]);
    let bcast_k = |x: &XlaOp| x.broadcast_in_dim(&[h, dk, dv], &[0, 1]);
    let bcast_v = |x: &XlaOp| x.broadcast_in_dim(&[h, dk, dv], &[0, 2]);

    // s <- s * exp(g_t)
    let s = (s * bcast_h(exp_g_t)?)?;
    // kv_mem = sum_dk(s * k_t)
    let kv_mem = (&s * bcast_k(k_t)?)?.reduce_sum(&[1], false)?;
    // delta = (v_t - kv_mem) * beta_t
    let beta_b = beta_t.broadcast_in_dim(&[h, dv], &[0])?;
    let delta = ((v_t - kv_mem)? * beta_b)?;
    // s <- s + outer(k_t, delta)
    let s = (s + (bcast_k(k_t)? * bcast_v(&delta)?)?)?;
    // out_t = sum_dk(s * q_t)
    let o_t = (&s * bcast_k(q_t)?)?.reduce_sum(&[1], false)?;
    Ok((s, o_t))
}

// The gated DeltaNet used on "linear_attention" layers.
struct GatedDeltaNet {
    in_proj_qkv: XlaOp,
    in_proj_z: XlaOp,
    in_proj_b: XlaOp,
    in_proj_a: XlaOp,
    conv1d: XlaOp,
    a_log: XlaOp,
    dt_bias: XlaOp,
    norm: XlaOp,
    out_proj: XlaOp,
}

impl GatedDeltaNet {
    fn new(vb: &VarBuilder, p: &str) -> Result<Self> {
        let value_dim = LIN_NUM_HEADS * LIN_VALUE_DIM;
        let in_proj_qkv = vb.var(&format!("{p}.in_proj_qkv.weight"), &[CONV_DIM, HIDDEN_SIZE])?;
        let in_proj_z = vb.var(&format!("{p}.in_proj_z.weight"), &[value_dim, HIDDEN_SIZE])?;
        let in_proj_b = vb.var(&format!("{p}.in_proj_b.weight"), &[LIN_NUM_HEADS, HIDDEN_SIZE])?;
        let in_proj_a = vb.var(&format!("{p}.in_proj_a.weight"), &[LIN_NUM_HEADS, HIDDEN_SIZE])?;
        let conv1d = vb.var(&format!("{p}.conv1d.weight"), &[CONV_DIM, 1, CONV_KERNEL_SIZE])?;
        let a_log = vb.var(&format!("{p}.A_log"), &[LIN_NUM_HEADS])?;
        let dt_bias = vb.var(&format!("{p}.dt_bias"), &[LIN_NUM_HEADS])?;
        let norm = vb.var(&format!("{p}.norm.weight"), &[LIN_VALUE_DIM])?;
        let out_proj = vb.var(&format!("{p}.out_proj.weight"), &[HIDDEN_SIZE, value_dim])?;
        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            a_log,
            dt_bias,
            norm,
            out_proj,
        })
    }

    // Depthwise causal conv1d (kernel 4, no bias) over the time dimension,
    // followed by a silu activation. x: [t, CONV_DIM].
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
        Ok(out.unwrap().silu()?)
    }

    // Split the post-conv activations into l2-normalized q/k and v, [t, h, d].
    // The values are converted to f32 as the whole recurrence is computed in
    // f32, as in the reference implementation.
    fn split_qkv(&self, mixed_qkv: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let b = mixed_qkv.builder();
        let h = LIN_NUM_HEADS;
        let (dk, dv) = (LIN_KEY_DIM, LIN_VALUE_DIM);
        let key_dim = h * dk;
        let value_dim = h * dv;
        let f32_at = |start: i64, stop: i64, d: i64| -> Result<XlaOp> {
            let x = mixed_qkv.slice_in_dim1(start, stop, 1)?.reshape(&[t, h, d])?;
            Ok(x.convert(PrimitiveType::F32)?)
        };
        let q = f32_at(0, key_dim, dk)?;
        let k = f32_at(key_dim, 2 * key_dim, dk)?;
        let v = f32_at(2 * key_dim, 2 * key_dim + value_dim, dv)?;
        // In-kernel l2 normalization of q/k plus query scaling.
        let q = (l2_norm(&q)? * b.c0(1f32 / (dk as f32).sqrt())?)?;
        let k = l2_norm(&k)?;
        Ok((q, k, v))
    }

    // The z/beta/exp(g) projections computed from the layer input, [t, ...].
    // z keeps the input dtype, beta and exp(g) are in f32 for the recurrence.
    fn gates(&self, x: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let h = LIN_NUM_HEADS;
        let z = linear(x, &self.in_proj_z)?.reshape(&[t, h, LIN_VALUE_DIM])?;
        let beta = linear(x, &self.in_proj_b)?.logistic()?.convert(PrimitiveType::F32)?;
        // g = -exp(A_log) * softplus(a + dt_bias), computed in f32.
        let a = linear(x, &self.in_proj_a)?.convert(PrimitiveType::F32)?;
        let dt_bias = self.dt_bias.convert(PrimitiveType::F32)?.broadcast_in_dim(&[t, h], &[1])?;
        let a_log_exp =
            self.a_log.convert(PrimitiveType::F32)?.exp()?.broadcast_in_dim(&[t, h], &[1])?;
        let exp_g = (a_log_exp.neg()? * softplus(&(a + dt_bias)?)?)?.exp()?;
        Ok((z, beta, exp_g))
    }

    // Gated rms-norm: norm(out) * weight * silu(z), with a plain (not
    // zero-centered) weight, followed by the output projection. out is in f32
    // (coming from the recurrence), the result uses the dtype of z.
    fn output(&self, out: &XlaOp, z: &XlaOp, t: i64) -> Result<XlaOp> {
        let b = out.builder();
        let h = LIN_NUM_HEADS;
        let dv = LIN_VALUE_DIM;
        let dt = z.ty()?;
        let mean2 = (out * out)?.reduce_mean(&[-1], true)?;
        let out_norm = (out * (mean2 + b.c0(RMS_NORM_EPS)?)?.rsqrt()?)?;
        let w = self.norm.convert(PrimitiveType::F32)?.broadcast_in_dim(&[t, h, dv], &[2])?;
        let z = z.convert(PrimitiveType::F32)?;
        let gated = ((out_norm * w)? * z.silu()?)?.convert(dt)?;
        linear(&gated.reshape(&[t, h * dv])?, &self.out_proj)
    }

    // The recurrent gated delta rule, computed sequentially over the first n
    // positions with a XLA while loop. q/k/v: [T, H, D], exp_g/beta: [T, H].
    // Returns the per-step outputs [T, H, D] (zeros beyond n) and the final
    // recurrent state [H, DK, DV].
    #[allow(clippy::too_many_arguments)]
    fn delta_rule(
        &self,
        builder: &XlaBuilder,
        q: &XlaOp,
        k: &XlaOp,
        v: &XlaOp,
        exp_g: &XlaOp,
        beta: &XlaOp,
        n: &XlaOp,
    ) -> Result<(XlaOp, XlaOp)> {
        let h = LIN_NUM_HEADS;
        let (dk, dv) = (LIN_KEY_DIM, LIN_VALUE_DIM);
        let state_shape = Shape::tuple(vec![
            Shape::array_with_type(ElementType::S32, vec![]), // t
            Shape::array::<f32>(vec![h, dk, dv]),             // recurrent state
            Shape::array::<f32>(vec![T, h, dv]),              // outputs
            Shape::array::<f32>(vec![T, h, dk]),              // q
            Shape::array::<f32>(vec![T, h, dk]),              // k
            Shape::array::<f32>(vec![T, h, dv]),              // v
            Shape::array::<f32>(vec![T, h]),                  // exp(g)
            Shape::array::<f32>(vec![T, h]),                  // beta
            Shape::array_with_type(ElementType::S32, vec![]), // n
        ]);
        let cond = {
            let b = XlaBuilder::new("cond");
            let state = b.parameter_s(0, &state_shape, "state")?;
            state.get_tuple_element(0)?.lt(&state.get_tuple_element(8)?)?.build()?
        };
        let body = {
            let b = XlaBuilder::new("body");
            let state = b.parameter_s(0, &state_shape, "state")?;
            let t = state.get_tuple_element(0)?;
            let s = state.get_tuple_element(1)?;
            let out = state.get_tuple_element(2)?;
            let q = state.get_tuple_element(3)?;
            let k = state.get_tuple_element(4)?;
            let v = state.get_tuple_element(5)?;
            let exp_g = state.get_tuple_element(6)?;
            let beta = state.get_tuple_element(7)?;
            let n = state.get_tuple_element(8)?;
            let zero = b.c0(0i32)?;
            let at_t = |x: &XlaOp, d: i64| -> Result<XlaOp> {
                Ok(x.dynamic_slice(&[&t, &zero, &zero], &[1, h, d])?.reshape(&[h, d])?)
            };
            let q_t = at_t(&q, dk)?;
            let k_t = at_t(&k, dk)?;
            let v_t = at_t(&v, dv)?;
            let exp_g_t = exp_g.dynamic_slice(&[&t, &zero], &[1, h])?.reshape(&[h])?;
            let beta_t = beta.dynamic_slice(&[&t, &zero], &[1, h])?.reshape(&[h])?;

            let (s, o_t) = delta_step(&s, &q_t, &k_t, &v_t, &exp_g_t, &beta_t)?;
            let out = out.dynamic_update_slice(&o_t.reshape(&[1, h, dv])?, &[&t, &zero, &zero])?;
            let t = (t + b.c0(1i32)?)?;
            b.tuple(&[t, s, out, q, k, v, exp_g, beta, n])?.build()?
        };
        let init_s = builder.c0(0f32)?.broadcast(&[h, dk, dv])?;
        let init_out = builder.c0(0f32)?.broadcast(&[T, h, dv])?;
        let init = builder.tuple(&[
            builder.c0(0i32)?,
            init_s,
            init_out,
            q.clone(),
            k.clone(),
            v.clone(),
            exp_g.clone(),
            beta.clone(),
            n.clone(),
        ])?;
        let result = XlaOp::while_(cond, body, init)?;
        Ok((result.get_tuple_element(2)?, result.get_tuple_element(1)?))
    }

    // Full-context forward. x: [T, HIDDEN]. Returns the outputs [T, HIDDEN],
    // the recurrent state after processing position last_pos, and the conv1d
    // inputs at positions last_pos-2..last_pos (zero padded on the left).
    fn forward_prefill(
        &self,
        builder: &XlaBuilder,
        x: &XlaOp,
        last_pos: &XlaOp,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let b = x.builder();
        let pre_conv = linear(x, &self.in_proj_qkv)?;
        let mixed_qkv = self.causal_conv(&pre_conv, T)?;
        let (q, k, v) = self.split_qkv(&mixed_qkv, T)?;
        let (z, beta, exp_g) = self.gates(x, T)?;
        let n = (last_pos + b.c0(1i32)?)?;
        let (out, s) = self.delta_rule(builder, &q, &k, &v, &exp_g, &beta, &n)?;
        let y = self.output(&out, &z, T)?;
        // The conv state holds the last CONV_KERNEL_SIZE-1 conv inputs; the
        // zero padding covers prompts shorter than the kernel and keeps the
        // dynamic slice in bounds.
        let zero_f = b.c0(0f32)?.convert(pre_conv.ty()?)?;
        let zero_i = b.c0(0i32)?;
        let padded = pre_conv.pad_in_dim(&zero_f, 0, CONV_KERNEL_SIZE - 1, 0)?;
        let conv_state = padded.dynamic_slice(&[&n, &zero_i], &[CONV_KERNEL_SIZE - 1, CONV_DIM])?;
        Ok((y, s, conv_state))
    }

    // Single position forward. x: [1, HIDDEN], s: [H, DK, DV],
    // conv_state: [CONV_KERNEL_SIZE-1, CONV_DIM]. Returns the output and the
    // updated states.
    fn forward_step(
        &self,
        x: &XlaOp,
        s: &XlaOp,
        conv_state: &XlaOp,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let h = LIN_NUM_HEADS;
        let (dk, dv) = (LIN_KEY_DIM, LIN_VALUE_DIM);
        let pre_conv = linear(x, &self.in_proj_qkv)?;
        let window = conv_state.concat_in_dim(&[&pre_conv], 0)?;
        let new_conv_state = window.slice_in_dim1(1, CONV_KERNEL_SIZE, 0)?;
        // out[c] = sum_j window[j, c] * w[c, j]
        let w = self.conv1d.reshape(&[CONV_DIM, CONV_KERNEL_SIZE])?.swap_dims(0, 1)?;
        let mixed_qkv = (window * w)?.reduce_sum(&[0], true)?.silu()?;
        let (q, k, v) = self.split_qkv(&mixed_qkv, 1)?;
        let (z, beta, exp_g) = self.gates(x, 1)?;
        let (s, o_t) = delta_step(
            &s.clone(),
            &q.reshape(&[h, dk])?,
            &k.reshape(&[h, dk])?,
            &v.reshape(&[h, dv])?,
            &exp_g.reshape(&[h])?,
            &beta.reshape(&[h])?,
        )?;
        let y = self.output(&o_t.reshape(&[1, h, dv])?, &z, 1)?;
        Ok((y, s, new_conv_state))
    }
}

// Full attention with GQA, partial rope, qk-norm, and a sigmoid output gate.
struct Attention {
    q_proj: XlaOp,
    k_proj: XlaOp,
    v_proj: XlaOp,
    o_proj: XlaOp,
    q_norm: XlaOp,
    k_norm: XlaOp,
}

impl Attention {
    fn new(vb: &VarBuilder, p: &str) -> Result<Self> {
        // The query projection produces both the queries and the output gate,
        // interleaved per head.
        let q_proj =
            vb.var(&format!("{p}.q_proj.weight"), &[2 * NUM_HEADS * HEAD_DIM, HIDDEN_SIZE])?;
        let k_proj =
            vb.var(&format!("{p}.k_proj.weight"), &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE])?;
        let v_proj =
            vb.var(&format!("{p}.v_proj.weight"), &[NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE])?;
        let o_proj = vb.var(&format!("{p}.o_proj.weight"), &[HIDDEN_SIZE, NUM_HEADS * HEAD_DIM])?;
        let q_norm = vb.var(&format!("{p}.q_norm.weight"), &[HEAD_DIM])?;
        let k_norm = vb.var(&format!("{p}.k_norm.weight"), &[HEAD_DIM])?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm })
    }

    // Rotate the first ROTARY_DIM dims of [t, h, HEAD_DIM] q or k values,
    // cos/sin: [t, ROTARY_DIM].
    fn apply_rope(&self, x: &XlaOp, cos: &XlaOp, sin: &XlaOp, t: i64, h: i64) -> Result<XlaOp> {
        let x_rot = x.slice_in_dim1(0, ROTARY_DIM, 2)?;
        let x_pass = x.slice_in_dim1(ROTARY_DIM, HEAD_DIM, 2)?;
        let x1 = x_rot.slice_in_dim1(0, ROTARY_DIM / 2, 2)?;
        let x2 = x_rot.slice_in_dim1(ROTARY_DIM / 2, ROTARY_DIM, 2)?;
        let rotated = x2.neg()?.concat_in_dim(&[&x1], 2)?;
        let dt = x.ty()?;
        let cos_b = cos.convert(dt)?.broadcast_in_dim(&[t, h, ROTARY_DIM], &[0, 2])?;
        let sin_b = sin.convert(dt)?.broadcast_in_dim(&[t, h, ROTARY_DIM], &[0, 2])?;
        let x_embed = ((x_rot * cos_b)? + (rotated * sin_b)?)?;
        Ok(x_embed.concat_in_dim(&[&x_pass], 2)?)
    }

    // The projections for t positions: post-rope q [t, h, D] and
    // k/v [t, kvh, D], plus the output gate [t, h*D].
    fn qkv_gate(
        &self,
        x: &XlaOp,
        cos: &XlaOp,
        sin: &XlaOp,
        t: i64,
    ) -> Result<(XlaOp, XlaOp, XlaOp, XlaOp)> {
        // Queries and gate, interleaved per head on the last dim.
        let q_and_gate = linear(x, &self.q_proj)?.reshape(&[t, NUM_HEADS, 2 * HEAD_DIM])?;
        let q = q_and_gate.slice_in_dim1(0, HEAD_DIM, 2)?;
        let gate = q_and_gate.slice_in_dim1(HEAD_DIM, 2 * HEAD_DIM, 2)?;
        let gate = gate.reshape(&[t, NUM_HEADS * HEAD_DIM])?;

        let q = rms_norm(&q, &self.q_norm)?;
        let k = linear(x, &self.k_proj)?.reshape(&[t, NUM_KV_HEADS, HEAD_DIM])?;
        let k = rms_norm(&k, &self.k_norm)?;
        let v = linear(x, &self.v_proj)?.reshape(&[t, NUM_KV_HEADS, HEAD_DIM])?;

        let q = self.apply_rope(&q, cos, sin, t, NUM_HEADS)?;
        let k = self.apply_rope(&k, cos, sin, t, NUM_KV_HEADS)?;
        Ok((q, k, v, gate))
    }

    // Attend from q [t, h, D] over k/v [s, kvh, D] with mask [t, s], apply the
    // sigmoid gate and the output projection.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        q: &XlaOp,
        k: &XlaOp,
        v: &XlaOp,
        gate: &XlaOp,
        mask: &XlaOp,
        t: i64,
        s: i64,
    ) -> Result<XlaOp> {
        let b = q.builder();
        // [t, h, d] -> [h, t, d], and repeat the kv heads for GQA.
        let q = q.swap_dims(0, 1)?;
        let repeat_kv = |x: &XlaOp| -> Result<XlaOp> {
            let x = x.swap_dims(0, 1)?;
            let groups = NUM_HEADS / NUM_KV_HEADS;
            Ok(x.broadcast_in_dim(&[NUM_KV_HEADS, groups, s, HEAD_DIM], &[0, 2, 3])?
                .reshape(&[NUM_HEADS, s, HEAD_DIM])?)
        };
        let k = repeat_kv(k)?;
        let v = repeat_kv(v)?;

        let dt = q.ty()?;
        let scale = b.c0(1f32 / (HEAD_DIM as f32).sqrt())?.convert(dt)?;
        let scores = (q.dot_general(&k, &[2], &[2], &[0], &[0])? * scale)?;
        let mask_b = mask.convert(dt)?.broadcast_in_dim(&[NUM_HEADS, t, s], &[1, 2])?;
        // The softmax is computed in f32 as in the reference implementation.
        let probs = (scores + mask_b)?.convert(PrimitiveType::F32)?.softmax(-1)?.convert(dt)?;
        let ctx = probs.dot_general(&v, &[2], &[1], &[0], &[0])?;
        let ctx = ctx.swap_dims(0, 1)?.reshape(&[t, NUM_HEADS * HEAD_DIM])?;

        let gated = (ctx * gate.logistic()?)?;
        linear(&gated, &self.o_proj)
    }

    // Full-context forward. Returns the outputs [T, HIDDEN] and the post-rope
    // k/v caches [T, kvh, D] (positions beyond the prompt hold garbage and get
    // overwritten by the decode steps).
    fn forward_prefill(
        &self,
        x: &XlaOp,
        cos: &XlaOp,
        sin: &XlaOp,
        mask: &XlaOp,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let (q, k, v, gate) = self.qkv_gate(x, cos, sin, T)?;
        let y = self.attend(&q, &k, &v, &gate, mask, T, T)?;
        Ok((y, k, v))
    }

    // Single position forward at position pos. cos/sin: [1, ROTARY_DIM] for
    // that position, mask: [1, T] allowing positions <= pos. Returns the
    // output and the updated k/v caches.
    #[allow(clippy::too_many_arguments)]
    fn forward_step(
        &self,
        x: &XlaOp,
        pos: &XlaOp,
        cos: &XlaOp,
        sin: &XlaOp,
        mask: &XlaOp,
        k_cache: &XlaOp,
        v_cache: &XlaOp,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let b = x.builder();
        let (q, k, v, gate) = self.qkv_gate(x, cos, sin, 1)?;
        let zero = b.c0(0i32)?;
        let k_cache = k_cache.dynamic_update_slice(&k, &[pos, &zero, &zero])?;
        let v_cache = v_cache.dynamic_update_slice(&v, &[pos, &zero, &zero])?;
        let y = self.attend(&q, &k_cache, &v_cache, &gate, mask, 1, T)?;
        Ok((y, k_cache, v_cache))
    }
}

enum Mixer {
    Attn(Attention),
    Lin(GatedDeltaNet),
}

struct DecoderLayer {
    input_ln: XlaOp,
    post_ln: XlaOp,
    mixer: Mixer,
    mlp: Mlp,
}

struct Model {
    embed: XlaOp,
    layers: Vec<DecoderLayer>,
    final_ln: XlaOp,
}

impl Model {
    // Weight declaration order must be identical between the prefill and the
    // decode builders as they share a single buffer list.
    fn new(vb: &VarBuilder) -> Result<Self> {
        let embed =
            vb.var("model.language_model.embed_tokens.weight", &[VOCAB_SIZE, HIDDEN_SIZE])?;
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for layer_idx in 0..NUM_LAYERS {
            let p = format!("model.language_model.layers.{layer_idx}");
            let input_ln = vb.var(&format!("{p}.input_layernorm.weight"), &[HIDDEN_SIZE])?;
            let post_ln =
                vb.var(&format!("{p}.post_attention_layernorm.weight"), &[HIDDEN_SIZE])?;
            let mixer = if (layer_idx + 1) % FULL_ATTENTION_INTERVAL == 0 {
                Mixer::Attn(Attention::new(vb, &format!("{p}.self_attn"))?)
            } else {
                Mixer::Lin(GatedDeltaNet::new(vb, &format!("{p}.linear_attn"))?)
            };
            let mlp = Mlp::new(vb, &format!("{p}.mlp"))?;
            layers.push(DecoderLayer { input_ln, post_ln, mixer, mlp });
        }
        let final_ln = vb.var("model.language_model.norm.weight", &[HIDDEN_SIZE])?;
        Ok(Self { embed, layers, final_ln })
    }
}

// Rotary embedding tables as constants, [T, ROTARY_DIM].
fn rope_tables(builder: &XlaBuilder) -> Result<(XlaOp, XlaOp)> {
    let mut cos_data = Vec::with_capacity((T * ROTARY_DIM) as usize);
    let mut sin_data = Vec::with_capacity((T * ROTARY_DIM) as usize);
    for t in 0..T {
        let mut freqs = Vec::with_capacity(ROTARY_DIM as usize);
        for i in 0..ROTARY_DIM / 2 {
            let inv_freq = 1f64 / ROPE_THETA.powf(2f64 * i as f64 / ROTARY_DIM as f64);
            freqs.push(t as f64 * inv_freq);
        }
        for i in 0..ROTARY_DIM as usize {
            let f = freqs[i % (ROTARY_DIM / 2) as usize];
            cos_data.push(f.cos() as f32);
            sin_data.push(f.sin() as f32);
        }
    }
    let cos = builder.c1(&cos_data)?.reshape(&[T, ROTARY_DIM])?;
    let sin = builder.c1(&sin_data)?.reshape(&[T, ROTARY_DIM])?;
    Ok((cos, sin))
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
    let (cos, sin) = rope_tables(builder)?;
    let mask = causal_mask(builder)?;

    let mut x = model.embed.take(&tokens, 0)?;
    let mut states = Vec::with_capacity(2 * NUM_LAYERS);
    for layer in model.layers.iter() {
        let residual = x.clone();
        let x_norm = rms_norm(&x, &layer.input_ln)?;
        let mixed = match &layer.mixer {
            Mixer::Attn(attn) => {
                let (y, k_cache, v_cache) = attn.forward_prefill(&x_norm, &cos, &sin, &mask)?;
                states.push(k_cache);
                states.push(v_cache);
                y
            }
            Mixer::Lin(delta_net) => {
                let (y, s, conv_state) = delta_net.forward_prefill(builder, &x_norm, &last_pos)?;
                states.push(s);
                states.push(conv_state);
                y
            }
        };
        let x_mid = (residual + mixed)?;
        x = (&x_mid + layer.mlp.forward(&rms_norm(&x_mid, &layer.post_ln)?)?)?;
    }

    let x = rms_norm(&x, &model.final_ln)?;
    // Only the logits for the last position are needed, the lm head reuses the
    // embedding weights (tie_word_embeddings).
    let zero = builder.c0(0i32)?;
    let x_last = x.dynamic_slice(&[&last_pos, &zero], &[1, HIDDEN_SIZE])?;
    let logits = linear(&x_last, &model.embed)?.convert(PrimitiveType::F32)?;
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
    let mut states = Vec::with_capacity(2 * NUM_LAYERS);
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        match &layer.mixer {
            Mixer::Attn(_) => {
                let dims = [T, NUM_KV_HEADS, HEAD_DIM];
                states.push(state_param(format!("layers.{layer_idx}.k_cache"), dtype, &dims)?);
                states.push(state_param(format!("layers.{layer_idx}.v_cache"), dtype, &dims)?);
            }
            Mixer::Lin(_) => {
                // The recurrent state is kept in f32, the conv state holds raw
                // conv inputs and uses the model dtype.
                let s_dims = [LIN_NUM_HEADS, LIN_KEY_DIM, LIN_VALUE_DIM];
                states.push(state_param(
                    format!("layers.{layer_idx}.state"),
                    ElementType::F32,
                    &s_dims,
                )?);
                let c_dims = [CONV_KERNEL_SIZE - 1, CONV_DIM];
                states.push(state_param(format!("layers.{layer_idx}.conv_state"), dtype, &c_dims)?);
            }
        }
    }

    let (cos, sin) = rope_tables(builder)?;
    let mask = causal_mask(builder)?;
    let zero = builder.c0(0i32)?;
    let cos = cos.dynamic_slice(&[&pos, &zero], &[1, ROTARY_DIM])?;
    let sin = sin.dynamic_slice(&[&pos, &zero], &[1, ROTARY_DIM])?;
    let mask = mask.dynamic_slice(&[&pos, &zero], &[1, T])?;

    let mut x = model.embed.take(&token, 0)?;
    let mut new_states = Vec::with_capacity(2 * NUM_LAYERS);
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let residual = x.clone();
        let x_norm = rms_norm(&x, &layer.input_ln)?;
        let mixed = match &layer.mixer {
            Mixer::Attn(attn) => {
                let (k_cache, v_cache) = (&states[2 * layer_idx], &states[2 * layer_idx + 1]);
                let (y, k_cache, v_cache) =
                    attn.forward_step(&x_norm, &pos, &cos, &sin, &mask, k_cache, v_cache)?;
                new_states.push(k_cache);
                new_states.push(v_cache);
                y
            }
            Mixer::Lin(delta_net) => {
                let (s, conv_state) = (&states[2 * layer_idx], &states[2 * layer_idx + 1]);
                let (y, s, conv_state) = delta_net.forward_step(&x_norm, s, conv_state)?;
                new_states.push(s);
                new_states.push(conv_state);
                y
            }
        };
        let x_mid = (residual + mixed)?;
        x = (&x_mid + layer.mlp.forward(&rms_norm(&x_mid, &layer.post_ln)?)?)?;
    }

    let x = rms_norm(&x, &model.final_ln)?;
    let logits = linear(&x, &model.embed)?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?;

    let mut outputs = vec![next_token];
    outputs.extend(new_states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// Download the tokenizer and weights from the hugging face hub, using the
// local cache if they have already been fetched.
fn hub_model_files() -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let client = hf_hub::HFClientSync::new()?;
    let repo = client.model("Qwen", "Qwen3.5-2B");
    let tokenizer = repo.download_file().filename("tokenizer.json").send()?;
    let weights =
        repo.download_file().filename("model.safetensors-00001-of-00001.safetensors").send()?;
    Ok((tokenizer, weights))
}

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
enum Dtype {
    F32,
    Bf16,
    F16,
}

impl Dtype {
    fn element_type(self) -> ElementType {
        match self {
            Self::F32 => ElementType::F32,
            Self::Bf16 => ElementType::Bf16,
            Self::F16 => ElementType::F16,
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Run on cpu rather than on gpu.
    #[arg(long)]
    cpu: bool,

    /// The dtype used for the weights and most of the computation, the norms,
    /// the attention softmax, and the DeltaNet recurrence always run in f32.
    #[arg(long, value_enum, default_value_t = Dtype::Bf16)]
    dtype: Dtype,

    /// The prompt used for generation.
    #[arg(long, default_value = "What is the capital of France? Answer in one word.")]
    prompt: String,

    /// The maximum number of tokens to generate.
    #[arg(long, default_value_t = 30)]
    sample_len: usize,

    /// Feed the raw prompt to the model rather than using the chat template.
    #[arg(long)]
    raw_prompt: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    xla::set_tf_min_log_level(xla::TfLogLevel::Warning);
    let client = if args.cpu { PjRtClient::cpu()? } else { PjRtClient::gpu(0.90, false)? };
    println!(
        "platform: {} {}, dtype: {:?}",
        client.platform_name(),
        client.platform_version(),
        args.dtype
    );

    let (tokenizer_path, weights_path) = hub_model_files()?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("cannot load tokenizer: {e}"))?;
    let prompt = if args.raw_prompt {
        args.prompt.clone()
    } else {
        format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", args.prompt)
    };
    let encoded = tokenizer.encode(prompt, false).map_err(|e| anyhow!("tokenizer error: {e}"))?;
    let mut tokens: Vec<i32> = encoded.get_ids().iter().map(|&t| t as i32).collect();
    println!("prompt has {} tokens", tokens.len());
    if tokens.is_empty() || tokens.len() >= CONTEXT_SIZE {
        anyhow::bail!("prompt length must be in [1, {}]", CONTEXT_SIZE - 1)
    }
    let stop_tokens: Vec<i32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s).map(|t| t as i32))
        .collect();

    let start = std::time::Instant::now();
    let prefill_builder = XlaBuilder::new("qwen35-prefill");
    let vb = VarBuilder::new(&prefill_builder, args.dtype.element_type());
    let prefill = build_prefill(&prefill_builder, &vb)?;
    let decode_builder = XlaBuilder::new("qwen35-decode");
    let decode_vb = VarBuilder::new(&decode_builder, args.dtype.element_type());
    let decode = build_decode(&decode_builder, &decode_vb)?;
    println!("built the computations in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let prefill_exe = client.compile(&prefill)?;
    let decode_exe = client.compile(&decode)?;
    println!("compiled the executables in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let weight_buffers = vb.load_buffers(&weights_path, &client)?;
    println!("loaded {} weights in {:?}", weight_buffers.len(), start.elapsed());

    let start = std::time::Instant::now();
    // Prefill: process the whole prompt, get the first token and the states.
    let mut padded = tokens.clone();
    padded.resize(CONTEXT_SIZE, 0);
    let token_buffer = client.buffer_from_host_buffer(&padded, &[CONTEXT_SIZE], None)?;
    let pos_buffer = client.buffer_from_host_buffer(&[tokens.len() as i32 - 1], &[], None)?;
    let mut inputs: Vec<&xla::PjRtBuffer> = vec![&token_buffer, &pos_buffer];
    inputs.extend(weight_buffers.iter());
    let mut outputs = prefill_exe
        .execute_b(&inputs)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no execution result"))?;
    let mut next_token = outputs[0].to_literal_sync()?.to_vec::<i32>()?[0];
    tokens.push(next_token);
    let mut generated = 1usize;

    // Decode: one token at a time, the state buffers stay on the device.
    while generated < args.sample_len
        && tokens.len() < CONTEXT_SIZE
        && !stop_tokens.contains(&next_token)
    {
        let token_buffer = client.buffer_from_host_buffer(&[next_token], &[1], None)?;
        let pos_buffer = client.buffer_from_host_buffer(&[tokens.len() as i32 - 1], &[], None)?;
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![&token_buffer, &pos_buffer];
        inputs.extend(weight_buffers.iter());
        inputs.extend(outputs[1..].iter());
        outputs = decode_exe
            .execute_b(&inputs)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no execution result"))?;
        next_token = outputs[0].to_literal_sync()?.to_vec::<i32>()?[0];
        tokens.push(next_token);
        generated += 1;
    }
    println!("generated {generated} tokens in {:?}", start.elapsed());
    println!("generated ids: {:?}", &tokens[tokens.len() - generated..]);

    let all_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&all_ids, false).map_err(|e| anyhow!("tokenizer error: {e}"))?;
    println!("----\n{text}\n----");
    Ok(())
}
