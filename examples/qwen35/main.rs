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

// Configuration shared by all the Qwen3.5 model sizes.
const FULL_ATTENTION_INTERVAL: usize = 4;
const VOCAB_SIZE: i64 = 248320;
const RMS_NORM_EPS: f32 = 1e-6;
// Full attention.
const HEAD_DIM: i64 = 256;
const ROPE_THETA: f64 = 1e7;
const ROTARY_DIM: i64 = 64; // partial_rotary_factor 0.25
                            // Linear attention (gated DeltaNet).
const LIN_KEY_DIM: i64 = 128;
const LIN_VALUE_DIM: i64 = 128;
const CONV_KERNEL_SIZE: i64 = 4;

const T: i64 = CONTEXT_SIZE as i64;

// Per-size configuration. The DeltaNet layers of the larger models use more
// value heads than key heads, the q/k heads are then repeated to match.
#[derive(Clone, Copy, Debug)]
struct Config {
    repo: &'static str,
    hidden_size: i64,
    intermediate_size: i64,
    num_layers: usize,
    num_heads: i64,
    num_kv_heads: i64,
    lin_num_k_heads: i64,
    lin_num_v_heads: i64,
    tie_embeddings: bool,
    num_shards: usize,
}

impl Config {
    fn conv_dim(&self) -> i64 {
        2 * self.lin_num_k_heads * LIN_KEY_DIM + self.lin_num_v_heads * LIN_VALUE_DIM
    }
}

const CONFIG_0_8B: Config = Config {
    repo: "Qwen/Qwen3.5-0.8B",
    hidden_size: 1024,
    intermediate_size: 3584,
    num_layers: 24,
    num_heads: 8,
    num_kv_heads: 2,
    lin_num_k_heads: 16,
    lin_num_v_heads: 16,
    tie_embeddings: true,
    num_shards: 1,
};

const CONFIG_2B: Config = Config {
    repo: "Qwen/Qwen3.5-2B",
    hidden_size: 2048,
    intermediate_size: 6144,
    num_layers: 24,
    num_heads: 8,
    num_kv_heads: 2,
    lin_num_k_heads: 16,
    lin_num_v_heads: 16,
    tie_embeddings: true,
    num_shards: 1,
};

const CONFIG_4B: Config = Config {
    repo: "Qwen/Qwen3.5-4B",
    hidden_size: 2560,
    intermediate_size: 9216,
    num_layers: 32,
    num_heads: 16,
    num_kv_heads: 4,
    lin_num_k_heads: 16,
    lin_num_v_heads: 32,
    tie_embeddings: true,
    num_shards: 2,
};

const CONFIG_9B: Config = Config {
    repo: "Qwen/Qwen3.5-9B",
    hidden_size: 4096,
    intermediate_size: 12288,
    num_layers: 32,
    num_heads: 16,
    num_kv_heads: 4,
    lin_num_k_heads: 16,
    lin_num_v_heads: 32,
    tie_embeddings: false,
    num_shards: 4,
};

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
    fn new(vb: &VarBuilder, p: &str, cfg: &Config) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate_proj = vb.var(&format!("{p}.gate_proj.weight"), &[i, h])?;
        let up_proj = vb.var(&format!("{p}.up_proj.weight"), &[i, h])?;
        let down_proj = vb.var(&format!("{p}.down_proj.weight"), &[h, i])?;
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
    h: i64,
) -> Result<(XlaOp, XlaOp)> {
    let (dk, dv) = (LIN_KEY_DIM, LIN_VALUE_DIM);
    let bcast_h = |x: &XlaOp| x.broadcast_in_dim(&[h, dk, dv], &[0]);
    let bcast_k = |x: &XlaOp| x.broadcast_in_dim(&[h, dk, dv], &[0, 1]);
    let bcast_v = |x: &XlaOp| x.broadcast_in_dim(&[h, dk, dv], &[0, 2]);

    // s <- s * exp(g_t)
    let s = (s * bcast_h(exp_g_t)?)?;
    // The two contractions of the delta rule against the decayed state,
    //   kv_mem = k_t . s and out_t = q_t . (s + outer(k_t, delta)),
    // are computed as a single [h, 2, dk] x [h, dk, dv] batched matmul with
    // out_t recovered as q_t . s + (q_t . k_t) * delta. A per-contraction
    // formulation (matvec or multiply-reduce) gets strength-reduced by XLA
    // into a reduce whose fusion compiles to a softmax-style triton kernel
    // with a single block per head, badly underutilizing the gpu.
    let q_r = q_t.reshape(&[h, 1, dk])?;
    let k_r = k_t.reshape(&[h, 1, dk])?;
    let qk = q_r.concat_in_dim(&[&k_r], 1)?;
    let qs_ks = qk.dot_general(&s, &[2], &[1], &[0], &[0])?;
    let qs = qs_ks.slice_in_dim1(0, 1, 1)?.reshape(&[h, dv])?;
    let ks = qs_ks.slice_in_dim1(1, 2, 1)?.reshape(&[h, dv])?;
    // delta = (v_t - kv_mem) * beta_t
    let beta_b = beta_t.broadcast_in_dim(&[h, dv], &[0])?;
    let delta = ((v_t - ks)? * beta_b)?;
    // s <- s + outer(k_t, delta)
    let s = (s + (bcast_k(k_t)? * bcast_v(&delta)?)?)?;
    // out_t = q_t . s (using the pre-update contraction, see above)
    let qk_dot = (q_t * k_t)?.reduce_sum(&[1], false)?;
    let o_t = (qs + (delta * qk_dot.broadcast_in_dim(&[h, dv], &[0])?)?)?;
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
    k_heads: i64,
    v_heads: i64,
    conv_dim: i64,
}

impl GatedDeltaNet {
    fn new(vb: &VarBuilder, p: &str, cfg: &Config) -> Result<Self> {
        let (kh, vh) = (cfg.lin_num_k_heads, cfg.lin_num_v_heads);
        let conv_dim = cfg.conv_dim();
        let hidden = cfg.hidden_size;
        let value_dim = vh * LIN_VALUE_DIM;
        let in_proj_qkv = vb.var(&format!("{p}.in_proj_qkv.weight"), &[conv_dim, hidden])?;
        let in_proj_z = vb.var(&format!("{p}.in_proj_z.weight"), &[value_dim, hidden])?;
        let in_proj_b = vb.var(&format!("{p}.in_proj_b.weight"), &[vh, hidden])?;
        let in_proj_a = vb.var(&format!("{p}.in_proj_a.weight"), &[vh, hidden])?;
        let conv1d = vb.var(&format!("{p}.conv1d.weight"), &[conv_dim, 1, CONV_KERNEL_SIZE])?;
        let a_log = vb.var(&format!("{p}.A_log"), &[vh])?;
        let dt_bias = vb.var(&format!("{p}.dt_bias"), &[vh])?;
        let norm = vb.var(&format!("{p}.norm.weight"), &[LIN_VALUE_DIM])?;
        let out_proj = vb.var(&format!("{p}.out_proj.weight"), &[hidden, value_dim])?;
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
            k_heads: kh,
            v_heads: vh,
            conv_dim,
        })
    }

    // Depthwise causal conv1d (kernel 4, no bias) over the time dimension,
    // followed by a silu activation. x: [t, conv_dim].
    fn causal_conv(&self, x: &XlaOp, t: i64) -> Result<XlaOp> {
        let b = x.builder();
        let w = self.conv1d.reshape(&[self.conv_dim, CONV_KERNEL_SIZE])?;
        let zero = b.c0(0f32)?.convert(x.ty()?)?;
        let mut out = None;
        for j in 0..CONV_KERNEL_SIZE {
            // shifted[i] = x[i - (k - 1 - j)]
            let shift = CONV_KERNEL_SIZE - 1 - j;
            let shifted = x.pad_in_dim(&zero, 0, shift, 0)?.slice_in_dim1(0, t, 0)?;
            let w_j = w.slice_in_dim1(j, j + 1, 1)?.reshape(&[self.conv_dim])?;
            let w_j = w_j.broadcast_in_dim(&[t, self.conv_dim], &[1])?;
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
    // f32, as in the reference implementation. When there are more value heads
    // than key heads, the q/k heads are repeated (interleaved) to match.
    fn split_qkv(&self, mixed_qkv: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let b = mixed_qkv.builder();
        let (kh, vh) = (self.k_heads, self.v_heads);
        let (dk, dv) = (LIN_KEY_DIM, LIN_VALUE_DIM);
        let key_dim = kh * dk;
        let value_dim = vh * dv;
        let f32_at = |start: i64, stop: i64, h: i64, d: i64| -> Result<XlaOp> {
            let x = mixed_qkv.slice_in_dim1(start, stop, 1)?.reshape(&[t, h, d])?;
            Ok(x.convert(PrimitiveType::F32)?)
        };
        let q = f32_at(0, key_dim, kh, dk)?;
        let k = f32_at(key_dim, 2 * key_dim, kh, dk)?;
        let v = f32_at(2 * key_dim, 2 * key_dim + value_dim, vh, dv)?;
        // In-kernel l2 normalization of q/k plus query scaling.
        let q = (l2_norm(&q)? * b.c0(1f32 / (dk as f32).sqrt())?)?;
        let k = l2_norm(&k)?;
        let repeat = |x: XlaOp| -> Result<XlaOp> {
            if kh == vh {
                return Ok(x);
            }
            Ok(x.broadcast_in_dim(&[t, kh, vh / kh, dk], &[0, 1, 3])?.reshape(&[t, vh, dk])?)
        };
        Ok((repeat(q)?, repeat(k)?, v))
    }

    // The z/beta/exp(g) projections computed from the layer input, [t, ...].
    // z keeps the input dtype, beta and exp(g) are in f32 for the recurrence.
    fn gates(&self, x: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let h = self.v_heads;
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
        let h = self.v_heads;
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
        let h = self.v_heads;
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

            let (s, o_t) = delta_step(&s, &q_t, &k_t, &v_t, &exp_g_t, &beta_t, h)?;
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
        let conv_state =
            padded.dynamic_slice(&[&n, &zero_i], &[CONV_KERNEL_SIZE - 1, self.conv_dim])?;
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
        let h = self.v_heads;
        let (dk, dv) = (LIN_KEY_DIM, LIN_VALUE_DIM);
        let pre_conv = linear(x, &self.in_proj_qkv)?;
        let window = conv_state.concat_in_dim(&[&pre_conv], 0)?;
        let new_conv_state = window.slice_in_dim1(1, CONV_KERNEL_SIZE, 0)?;
        // out[c] = sum_j window[j, c] * w[c, j]
        let w = self.conv1d.reshape(&[self.conv_dim, CONV_KERNEL_SIZE])?.swap_dims(0, 1)?;
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
            h,
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
    n_heads: i64,
    n_kv_heads: i64,
}

impl Attention {
    fn new(vb: &VarBuilder, p: &str, cfg: &Config) -> Result<Self> {
        let (nh, nkv) = (cfg.num_heads, cfg.num_kv_heads);
        let hidden = cfg.hidden_size;
        // The query projection produces both the queries and the output gate,
        // interleaved per head.
        let q_proj = vb.var(&format!("{p}.q_proj.weight"), &[2 * nh * HEAD_DIM, hidden])?;
        let k_proj = vb.var(&format!("{p}.k_proj.weight"), &[nkv * HEAD_DIM, hidden])?;
        let v_proj = vb.var(&format!("{p}.v_proj.weight"), &[nkv * HEAD_DIM, hidden])?;
        let o_proj = vb.var(&format!("{p}.o_proj.weight"), &[hidden, nh * HEAD_DIM])?;
        let q_norm = vb.var(&format!("{p}.q_norm.weight"), &[HEAD_DIM])?;
        let k_norm = vb.var(&format!("{p}.k_norm.weight"), &[HEAD_DIM])?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, n_heads: nh, n_kv_heads: nkv })
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
        let (nh, nkv) = (self.n_heads, self.n_kv_heads);
        // Queries and gate, interleaved per head on the last dim.
        let q_and_gate = linear(x, &self.q_proj)?.reshape(&[t, nh, 2 * HEAD_DIM])?;
        let q = q_and_gate.slice_in_dim1(0, HEAD_DIM, 2)?;
        let gate = q_and_gate.slice_in_dim1(HEAD_DIM, 2 * HEAD_DIM, 2)?;
        let gate = gate.reshape(&[t, nh * HEAD_DIM])?;

        let q = rms_norm(&q, &self.q_norm)?;
        let k = linear(x, &self.k_proj)?.reshape(&[t, nkv, HEAD_DIM])?;
        let k = rms_norm(&k, &self.k_norm)?;
        let v = linear(x, &self.v_proj)?.reshape(&[t, nkv, HEAD_DIM])?;

        let q = self.apply_rope(&q, cos, sin, t, nh)?;
        let k = self.apply_rope(&k, cos, sin, t, nkv)?;
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
        let (nh, nkv) = (self.n_heads, self.n_kv_heads);
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
        // The softmax is computed in f32 as in the reference implementation.
        let probs = (scores + mask_b)?.convert(PrimitiveType::F32)?.softmax(-1)?.convert(dt)?;
        let ctx = probs.dot_general(&v, &[2], &[1], &[0], &[0])?;
        let ctx = ctx.swap_dims(0, 1)?.reshape(&[t, nh * HEAD_DIM])?;

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
    lm_head: Option<XlaOp>,
}

impl Model {
    // Weight declaration order must be identical between the prefill and the
    // decode builders as they share a single buffer list.
    fn new(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let embed = vb.var("model.language_model.embed_tokens.weight", &[VOCAB_SIZE, hidden])?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let p = format!("model.language_model.layers.{layer_idx}");
            let input_ln = vb.var(&format!("{p}.input_layernorm.weight"), &[hidden])?;
            let post_ln = vb.var(&format!("{p}.post_attention_layernorm.weight"), &[hidden])?;
            let mixer = if (layer_idx + 1) % FULL_ATTENTION_INTERVAL == 0 {
                Mixer::Attn(Attention::new(vb, &format!("{p}.self_attn"), cfg)?)
            } else {
                Mixer::Lin(GatedDeltaNet::new(vb, &format!("{p}.linear_attn"), cfg)?)
            };
            let mlp = Mlp::new(vb, &format!("{p}.mlp"), cfg)?;
            layers.push(DecoderLayer { input_ln, post_ln, mixer, mlp });
        }
        let final_ln = vb.var("model.language_model.norm.weight", &[hidden])?;
        let lm_head = if cfg.tie_embeddings {
            None
        } else {
            Some(vb.var("lm_head.weight", &[VOCAB_SIZE, hidden])?)
        };
        Ok(Self { embed, layers, final_ln, lm_head })
    }

    fn lm_head(&self) -> &XlaOp {
        self.lm_head.as_ref().unwrap_or(&self.embed)
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
fn build_prefill(builder: &XlaBuilder, vb: &VarBuilder, cfg: &Config) -> Result<XlaComputation> {
    let tokens = builder.parameter(0, ElementType::S32, &[T], "tokens")?;
    let last_pos = builder.parameter(1, ElementType::S32, &[], "last_pos")?;
    let model = Model::new(vb, cfg)?;
    let (cos, sin) = rope_tables(builder)?;
    let mask = causal_mask(builder)?;

    let mut x = model.embed.take(&tokens, 0)?;
    let mut states = Vec::with_capacity(2 * cfg.num_layers);
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
    // Only the logits for the last position are needed. On the models with
    // tie_word_embeddings the lm head reuses the embedding weights.
    let zero = builder.c0(0i32)?;
    let x_last = x.dynamic_slice(&[&last_pos, &zero], &[1, cfg.hidden_size])?;
    let logits = linear(&x_last, model.lm_head())?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?;

    let mut outputs = vec![next_token];
    outputs.extend(states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// The decode computation: a single token at a given position plus the
// per-layer states in, next token plus the updated states out. The states are
// passed as parameters after the weights so that the weight parameter indices
// match the prefill computation.
fn build_decode(builder: &XlaBuilder, vb: &VarBuilder, cfg: &Config) -> Result<XlaComputation> {
    let token = builder.parameter(0, ElementType::S32, &[1], "token")?;
    let pos = builder.parameter(1, ElementType::S32, &[], "pos")?;
    let model = Model::new(vb, cfg)?;

    let mut param_idx = (NUM_NON_WEIGHT_ARGS + vb.num_vars()) as i64;
    let mut state_param = |name: String, ty: ElementType, dims: &[i64]| -> Result<XlaOp> {
        let op = builder.parameter(param_idx, ty, dims, &name)?;
        param_idx += 1;
        Ok(op)
    };
    let dtype = vb.dtype();
    let mut states = Vec::with_capacity(2 * cfg.num_layers);
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        match &layer.mixer {
            Mixer::Attn(_) => {
                let dims = [T, cfg.num_kv_heads, HEAD_DIM];
                states.push(state_param(format!("layers.{layer_idx}.k_cache"), dtype, &dims)?);
                states.push(state_param(format!("layers.{layer_idx}.v_cache"), dtype, &dims)?);
            }
            Mixer::Lin(_) => {
                // The recurrent state is kept in f32, the conv state holds raw
                // conv inputs and uses the model dtype.
                let s_dims = [cfg.lin_num_v_heads, LIN_KEY_DIM, LIN_VALUE_DIM];
                states.push(state_param(
                    format!("layers.{layer_idx}.state"),
                    ElementType::F32,
                    &s_dims,
                )?);
                let c_dims = [CONV_KERNEL_SIZE - 1, cfg.conv_dim()];
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
    let mut new_states = Vec::with_capacity(2 * cfg.num_layers);
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
    let logits = linear(&x, model.lm_head())?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?;

    let mut outputs = vec![next_token];
    outputs.extend(new_states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// hf-hub does not report download progress unless a handler is registered, so
// a first run (which fetches several GB of weights) otherwise looks frozen.
// This handler prints a single self-updating percentage line per file to
// stderr, covering both the plain per-file stream and the xet aggregate path
// (the safetensors shards go through xet).
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
                if total > 0 {
                    self.done.store(total, Ordering::Relaxed);
                }
                self.render();
                eprintln!();
            }
        }
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", UNITS[i])
}

// Download the tokenizer and weight shards from the hugging face hub, using
// the local cache if they have already been fetched.
fn hub_model_files(cfg: &Config) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let client = hf_hub::HFClientSync::new()?;
    let (owner, name) = cfg.repo.split_once('/').ok_or_else(|| anyhow!("invalid repo"))?;
    let repo = client.model(owner, name);
    let tokenizer = repo
        .download_file()
        .filename("tokenizer.json")
        .progress(DownloadProgress::new("tokenizer.json"))
        .send()?;
    let mut weights = Vec::with_capacity(cfg.num_shards);
    for shard in 1..=cfg.num_shards {
        let filename =
            format!("model.safetensors-{:05}-of-{:05}.safetensors", shard, cfg.num_shards);
        weights.push(
            repo.download_file()
                .filename(&filename)
                .progress(DownloadProgress::new(filename.as_str()))
                .send()?,
        );
    }
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

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
enum Which {
    #[value(name = "0.8b")]
    W0_8b,
    #[value(name = "2b")]
    W2b,
    #[value(name = "4b")]
    W4b,
    #[value(name = "9b")]
    W9b,
}

impl Which {
    fn config(self) -> &'static Config {
        match self {
            Self::W0_8b => &CONFIG_0_8B,
            Self::W2b => &CONFIG_2B,
            Self::W4b => &CONFIG_4B,
            Self::W9b => &CONFIG_9B,
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Run on cpu rather than on gpu.
    #[arg(long)]
    cpu: bool,

    /// The model size to run.
    #[arg(long, value_enum, default_value_t = Which::W2b)]
    which: Which,

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
    let cfg = args.which.config();
    let client = if args.cpu { PjRtClient::cpu()? } else { PjRtClient::gpu(0.90, false)? };
    println!(
        "platform: {} {}, model: {}, dtype: {:?}",
        client.platform_name(),
        client.platform_version(),
        cfg.repo,
        args.dtype
    );

    let (tokenizer_path, weights_paths) = hub_model_files(cfg)?;
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
    let prefill = build_prefill(&prefill_builder, &vb, cfg)?;
    let decode_builder = XlaBuilder::new("qwen35-decode");
    let decode_vb = VarBuilder::new(&decode_builder, args.dtype.element_type());
    let decode = build_decode(&decode_builder, &decode_vb, cfg)?;
    println!("built the computations in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let prefill_exe = client.compile(&prefill)?;
    let decode_exe = client.compile(&decode)?;
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
    // Sync on the first generated token so that the prefill can be timed
    // separately from the decode loop. The token stays on the device and gets
    // re-read (cheaply) in the first loop iteration below.
    prefill_outputs[0].to_literal_sync()?;
    println!("prefill ({} tokens) in {:?}", tokens.len(), start.elapsed());
    let start = std::time::Instant::now();

    // Decode: one token at a time, the state buffers stay on the device. The
    // generated token is also chained on the device: the next step is
    // dispatched before the current token is read back, so that the host to
    // device round-trip and the dispatch overlap with the device execution.
    // The step dispatched when a stop token shows up is wasted, but harmless.
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
    // The first token comes from the prefill, so the decode loop ran
    // generated - 1 steps.
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
