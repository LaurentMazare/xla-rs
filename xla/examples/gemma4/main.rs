// Inference example for the Gemma 4 E2B and E4B models,
// https://huggingface.co/google/gemma-4-E2B-it (text model part only).
//
// The Gemma 4 E models are MatFormer-style hybrids: most decoder layers use
// sliding-window attention (window 512, four out of five layers on E2B, five
// out of six on E4B) and the remaining ones use full attention with a wider
// 512 head dim and a different rope configuration. The last layers (20 on
// E2B, 18 on E4B) do not have their own k/v projections and instead reuse
// the keys/values of the last non-shared layer of the same attention type
// (and use a double-width mlp on E2B). Each token also gets a per-layer
// embedding (PLE) that is mixed into the residual stream after the mlp of
// every layer.
//
// The PLE table is the "E" in the model names: it accounts for around half
// of the parameters (2.4B on E2B, 2.9B on E4B) but is only ever read one row
// per token, so by default it is kept in host memory and the rows for the
// current tokens are gathered on the cpu and passed to the computations as
// an input. This keeps the device weights at 4.5GB for E2B and 9.2GB for
// E4B, which is how E4B fits on a 16GB gpu in bf16 (everything runs in
// bf16). The --ple-on-device flag keeps the table in device memory instead,
// which lets the decode loop chain the generated token on the device rather
// than syncing it back to the host at every step.
//
// The structure mirrors the qwen35 example: a prefill computation processes
// the whole padded context and returns the first token plus the per-layer k/v
// caches, and a decode computation processes a single position. As the fixed
// 128 token context is smaller than the 512 sliding window, sliding and full
// attention use the same causal mask.
use anyhow::{anyhow, Result};
use clap::Parser;

extern crate xla;
use xla::{ElementType, PjRtClient, PrimitiveType, XlaBuilder, XlaComputation, XlaOp};

mod var_store;
use var_store::VarBuilder;

// Fixed context size the computations get compiled for, also the kv-cache
// length. Must stay <= sliding_window (512) so that sliding attention
// degenerates to causal attention.
const CONTEXT_SIZE: usize = 128;
const T: i64 = CONTEXT_SIZE as i64;

// Dims shared between the E2B and E4B text configs.
const NUM_HEADS: i64 = 8;
const HEAD_DIM: i64 = 256; // sliding attention layers
const GLOBAL_HEAD_DIM: i64 = 512; // full attention layers
const VOCAB_SIZE: i64 = 262144;
const PLE_DIM: i64 = 256; // hidden_size_per_layer_input
const RMS_NORM_EPS: f32 = 1e-6;
const FINAL_LOGIT_SOFTCAP: f32 = 30.0;
// Rope: sliding layers use the default parametrization with theta 1e4, full
// attention layers use a "proportional" one with theta 1e6 where only the
// first partial_rotary_factor=0.25 fraction of the angles is rotated (the
// remaining frequencies are zero, i.e. pass-through).
const LOCAL_ROPE_THETA: f64 = 1e4;
const GLOBAL_ROPE_THETA: f64 = 1e6;
const GLOBAL_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

// The dims that differ between the model sizes.
struct Config {
    repo: &'static str,
    hidden_size: i64,
    num_layers: usize,
    num_kv_heads: i64,
    intermediate_size: i64,
    // The kv-shared layers use a double-width mlp on E2B but not on E4B.
    double_wide_shared_mlp: bool,
    // Every n-th layer uses full attention, the others sliding attention.
    full_attention_interval: usize,
    // Layers from this index on reuse the k/v of the last non-shared layer
    // of the same attention type (the checkpoint contains k/v weights for
    // them, unused).
    first_kv_shared_layer: usize,
}

const CONFIG_E2B: Config = Config {
    repo: "gemma-4-E2B-it",
    hidden_size: 1536,
    num_layers: 35,
    num_kv_heads: 1,
    intermediate_size: 6144,
    double_wide_shared_mlp: true,
    full_attention_interval: 5,
    first_kv_shared_layer: 15,
};

const CONFIG_E4B: Config = Config {
    repo: "gemma-4-E4B-it",
    hidden_size: 2560,
    num_layers: 42,
    num_kv_heads: 2,
    intermediate_size: 10240,
    double_wide_shared_mlp: false,
    full_attention_interval: 6,
    first_kv_shared_layer: 24,
};

impl Config {
    fn is_full_attention(&self, layer_idx: usize) -> bool {
        (layer_idx + 1).is_multiple_of(self.full_attention_interval)
    }

    fn is_kv_shared(&self, layer_idx: usize) -> bool {
        layer_idx >= self.first_kv_shared_layer
    }

    // The last non-shared layer of the same attention type, providing the k/v
    // states for the kv-shared layers.
    fn kv_donor(&self, layer_idx: usize) -> usize {
        let mut donor = 0;
        for i in 0..self.first_kv_shared_layer {
            if self.is_full_attention(i) == self.is_full_attention(layer_idx) {
                donor = i
            }
        }
        donor
    }

    fn head_dim(&self, layer_idx: usize) -> i64 {
        if self.is_full_attention(layer_idx) {
            GLOBAL_HEAD_DIM
        } else {
            HEAD_DIM
        }
    }
}

fn linear(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    // x: [..., in], w: [out, in] -> [..., out]
    let x_rank = x.rank()? as i64;
    Ok(x.dot_general(w, &[x_rank - 1], &[1], &[], &[])?)
}

// Gemma rms-norm: computed in f32, scaled by the plain weight (not the
// zero-centered 1 + weight form used by qwen), and cast back to the input
// dtype. The weight is applied over the last dimension.
fn rms_norm(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let mean2 = ((&x * &x)?.reduce_mean(&[-1], true)? + b.c0(RMS_NORM_EPS)?)?;
    let x_norm = (&x * mean2.rsqrt()?)?;
    let rank = x.rank()? as i64;
    let w =
        w.convert(PrimitiveType::F32)?.broadcast_in_dim(x.array_shape()?.dims(), &[rank - 1])?;
    Ok((x_norm * w)?.convert(dt)?)
}

// Same without a learned scale, used for the values in the attention layers.
fn rms_norm_no_scale(x: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let mean2 = ((&x * &x)?.reduce_mean(&[-1], true)? + b.c0(RMS_NORM_EPS)?)?;
    Ok((&x * mean2.rsqrt()?)?.convert(dt)?)
}

// gelu with the tanh approximation (gelu_pytorch_tanh), computed in f32 and
// rounded back to the input dtype so that it matches the single-op torch
// semantics on bf16 inputs.
fn gelu_tanh(x: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let x3 = ((&x * &x)? * &x)?;
    let inner = ((&x + (x3 * b.c0(0.044715f32)?)?)? * b.c0(0.797_884_6_f32)?)?;
    let y = ((inner.tanh()? + b.c0(1f32)?)? * (x * b.c0(0.5f32)?)?)?;
    Ok(y.convert(dt)?)
}

struct Mlp {
    gate_proj: XlaOp,
    up_proj: XlaOp,
    down_proj: XlaOp,
}

impl Mlp {
    fn new(vb: &VarBuilder, p: &str, layer_idx: usize, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = if cfg.double_wide_shared_mlp && cfg.is_kv_shared(layer_idx) {
            2 * cfg.intermediate_size
        } else {
            cfg.intermediate_size
        };
        let gate_proj = vb.var(&format!("{p}.gate_proj.weight"), &[i, h])?;
        let up_proj = vb.var(&format!("{p}.up_proj.weight"), &[i, h])?;
        let down_proj = vb.var(&format!("{p}.down_proj.weight"), &[h, i])?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let gated = (gelu_tanh(&linear(x, &self.gate_proj)?)? * linear(x, &self.up_proj)?)?;
        linear(&gated, &self.down_proj)
    }
}

// Attention with one or two k/v heads (E2B/E4B) broadcast over the 8 query
// heads, rms norms on q/k (learned scale) and v (no scale), rope applied
// after the norms, and no 1/sqrt(d) scaling (the q norm takes care of the
// scale). The kv-shared layers only have q/o projections.
struct Attention {
    q_proj: XlaOp,
    q_norm: XlaOp,
    o_proj: XlaOp,
    kv: Option<(XlaOp, XlaOp, XlaOp)>, // k_proj, k_norm, v_proj
    hd: i64,
    kvh: i64,
}

impl Attention {
    fn new(vb: &VarBuilder, p: &str, layer_idx: usize, cfg: &Config) -> Result<Self> {
        let hd = cfg.head_dim(layer_idx);
        let (h, kvh) = (cfg.hidden_size, cfg.num_kv_heads);
        let q_proj = vb.var(&format!("{p}.q_proj.weight"), &[NUM_HEADS * hd, h])?;
        let q_norm = vb.var(&format!("{p}.q_norm.weight"), &[hd])?;
        let o_proj = vb.var(&format!("{p}.o_proj.weight"), &[h, NUM_HEADS * hd])?;
        let kv = if cfg.is_kv_shared(layer_idx) {
            None
        } else {
            let k_proj = vb.var(&format!("{p}.k_proj.weight"), &[kvh * hd, h])?;
            let k_norm = vb.var(&format!("{p}.k_norm.weight"), &[hd])?;
            let v_proj = vb.var(&format!("{p}.v_proj.weight"), &[kvh * hd, h])?;
            Some((k_proj, k_norm, v_proj))
        };
        Ok(Self { q_proj, q_norm, o_proj, kv, hd, kvh })
    }

    // Rope over the full head dim: cos/sin are [t, hd] tables (with cos=1,
    // sin=0 on the non-rotated fraction of the full attention layers).
    fn apply_rope(&self, x: &XlaOp, cos: &XlaOp, sin: &XlaOp, t: i64, h: i64) -> Result<XlaOp> {
        let hd = self.hd;
        let x1 = x.slice_in_dim1(0, hd / 2, 2)?;
        let x2 = x.slice_in_dim1(hd / 2, hd, 2)?;
        let rotated = x2.neg()?.concat_in_dim(&[&x1], 2)?;
        let dt = x.ty()?;
        let cos_b = cos.convert(dt)?.broadcast_in_dim(&[t, h, hd], &[0, 2])?;
        let sin_b = sin.convert(dt)?.broadcast_in_dim(&[t, h, hd], &[0, 2])?;
        Ok(((x * cos_b)? + (rotated * sin_b)?)?)
    }

    // Post-norm post-rope queries [t, h, hd] for t positions.
    fn q(&self, x: &XlaOp, cos: &XlaOp, sin: &XlaOp, t: i64) -> Result<XlaOp> {
        let q = linear(x, &self.q_proj)?.reshape(&[t, NUM_HEADS, self.hd])?;
        let q = rms_norm(&q, &self.q_norm)?;
        self.apply_rope(&q, cos, sin, t, NUM_HEADS)
    }

    // Post-norm post-rope keys and post-norm values [t, kvh, hd], only
    // available on the non-shared layers.
    fn kv(&self, x: &XlaOp, cos: &XlaOp, sin: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp)> {
        let (k_proj, k_norm, v_proj) = self.kv.as_ref().ok_or_else(|| anyhow!("kv on shared"))?;
        let k = linear(x, k_proj)?.reshape(&[t, self.kvh, self.hd])?;
        let k = rms_norm(&k, k_norm)?;
        let k = self.apply_rope(&k, cos, sin, t, self.kvh)?;
        let v = linear(x, v_proj)?.reshape(&[t, self.kvh, self.hd])?;
        let v = rms_norm_no_scale(&v)?;
        Ok((k, v))
    }

    // Attend from q [t, h, hd] over k/v [s, kvh, hd] with mask [t, s]. The
    // attention scores are not scaled (scaling factor 1), the softmax runs in
    // f32 as in the reference implementation.
    fn attend(
        &self,
        q: &XlaOp,
        k: &XlaOp,
        v: &XlaOp,
        mask: &XlaOp,
        t: i64,
        s: i64,
    ) -> Result<XlaOp> {
        let hd = self.hd;
        let kvh = self.kvh;
        // [t, h, d] -> [h, t, d], and broadcast the kv heads over the groups.
        let q = q.swap_dims(0, 1)?;
        let repeat_kv = |x: &XlaOp| -> Result<XlaOp> {
            let x = x.swap_dims(0, 1)?;
            let groups = NUM_HEADS / kvh;
            Ok(x.broadcast_in_dim(&[kvh, groups, s, hd], &[0, 2, 3])?
                .reshape(&[NUM_HEADS, s, hd])?)
        };
        let k = repeat_kv(k)?;
        let v = repeat_kv(v)?;
        let dt = q.ty()?;
        let scores = q.dot_general(&k, &[2], &[2], &[0], &[0])?;
        let mask_b = mask.convert(dt)?.broadcast_in_dim(&[NUM_HEADS, t, s], &[1, 2])?;
        let probs = (scores + mask_b)?.convert(PrimitiveType::F32)?.softmax(-1)?.convert(dt)?;
        let ctx = probs.dot_general(&v, &[2], &[1], &[0], &[0])?;
        let ctx = ctx.swap_dims(0, 1)?.reshape(&[t, NUM_HEADS * hd])?;
        linear(&ctx, &self.o_proj)
    }
}

struct DecoderLayer {
    input_ln: XlaOp,
    post_attn_ln: XlaOp,
    pre_ff_ln: XlaOp,
    post_ff_ln: XlaOp,
    attn: Attention,
    mlp: Mlp,
    ple_gate: XlaOp,
    ple_proj: XlaOp,
    ple_norm: XlaOp,
    layer_scalar: XlaOp,
    h: i64,
}

impl DecoderLayer {
    fn new(vb: &VarBuilder, p: &str, layer_idx: usize, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let input_ln = vb.var(&format!("{p}.input_layernorm.weight"), &[h])?;
        let post_attn_ln = vb.var(&format!("{p}.post_attention_layernorm.weight"), &[h])?;
        let pre_ff_ln = vb.var(&format!("{p}.pre_feedforward_layernorm.weight"), &[h])?;
        let post_ff_ln = vb.var(&format!("{p}.post_feedforward_layernorm.weight"), &[h])?;
        let attn = Attention::new(vb, &format!("{p}.self_attn"), layer_idx, cfg)?;
        let mlp = Mlp::new(vb, &format!("{p}.mlp"), layer_idx, cfg)?;
        let ple_gate = vb.var(&format!("{p}.per_layer_input_gate.weight"), &[PLE_DIM, h])?;
        let ple_proj = vb.var(&format!("{p}.per_layer_projection.weight"), &[h, PLE_DIM])?;
        let ple_norm = vb.var(&format!("{p}.post_per_layer_input_norm.weight"), &[h])?;
        let layer_scalar = vb.var(&format!("{p}.layer_scalar"), &[1])?;
        Ok(Self {
            input_ln,
            post_attn_ln,
            pre_ff_ln,
            post_ff_ln,
            attn,
            mlp,
            ple_gate,
            ple_proj,
            ple_norm,
            layer_scalar,
            h,
        })
    }

    // Everything after the attention: sandwich-norm residual adds, the mlp,
    // the per-layer embedding block, and the final layer scalar.
    // x/attn_out/per_layer_input: [t, ...].
    fn post_attention(
        &self,
        x: &XlaOp,
        attn_out: &XlaOp,
        ple_input: &XlaOp,
        t: i64,
    ) -> Result<XlaOp> {
        let x = (x + rms_norm(attn_out, &self.post_attn_ln)?)?;
        let mlp_out = self.mlp.forward(&rms_norm(&x, &self.pre_ff_ln)?)?;
        let x = (&x + rms_norm(&mlp_out, &self.post_ff_ln)?)?;
        // PLE: gate the hidden state, multiply with the per-layer input and
        // project back to the residual stream.
        let g = gelu_tanh(&linear(&x, &self.ple_gate)?)?;
        let y = linear(&(g * ple_input)?, &self.ple_proj)?;
        let x = (&x + rms_norm(&y, &self.ple_norm)?)?;
        let scalar = self.layer_scalar.reshape(&[1])?.broadcast_in_dim(&[t, self.h], &[1])?;
        Ok((x * scalar)?)
    }
}

struct Model {
    embed: XlaOp,
    // The on-device per-layer embedding table, None when it is kept on the
    // host and the gathered rows come in through a computation parameter.
    embed_per_layer: Option<XlaOp>,
    ple_projection: XlaOp,
    ple_projection_norm: XlaOp,
    layers: Vec<DecoderLayer>,
    final_ln: XlaOp,
    hidden_size: i64,
    num_layers: usize,
}

impl Model {
    // Weight declaration order must be identical between the prefill and the
    // decode builders as they share a single buffer list.
    fn new(vb: &VarBuilder, cfg: &Config, ple_on_device: bool) -> Result<Self> {
        let (h, n) = (cfg.hidden_size, cfg.num_layers);
        let pre = "model.language_model";
        let embed = vb.var(&format!("{pre}.embed_tokens.weight"), &[VOCAB_SIZE, h])?;
        let embed_per_layer = if ple_on_device {
            Some(vb.var(
                &format!("{pre}.embed_tokens_per_layer.weight"),
                &[VOCAB_SIZE, n as i64 * PLE_DIM],
            )?)
        } else {
            None
        };
        let ple_projection =
            vb.var(&format!("{pre}.per_layer_model_projection.weight"), &[n as i64 * PLE_DIM, h])?;
        let ple_projection_norm =
            vb.var(&format!("{pre}.per_layer_projection_norm.weight"), &[PLE_DIM])?;
        let mut layers = Vec::with_capacity(n);
        for layer_idx in 0..n {
            let p = format!("{pre}.layers.{layer_idx}");
            layers.push(DecoderLayer::new(vb, &p, layer_idx, cfg)?);
        }
        let final_ln = vb.var(&format!("{pre}.norm.weight"), &[h])?;
        Ok(Self {
            embed,
            embed_per_layer,
            ple_projection,
            ple_projection_norm,
            layers,
            final_ln,
            hidden_size: h,
            num_layers: n,
        })
    }

    // Scaled token embeddings [t, HIDDEN]. The sqrt(hidden) scale is rounded
    // to bf16 as in the reference implementation.
    fn embed(&self, tokens: &XlaOp, dt: PrimitiveType) -> Result<XlaOp> {
        let b = tokens.builder();
        let scale = b.c0((self.hidden_size as f32).sqrt())?.convert(dt)?;
        let x = self.embed.take(tokens, 0)?;
        let scale = scale.broadcast_in_dim(x.array_shape()?.dims(), &[])?;
        Ok((x * scale)?)
    }

    // Combined per-layer inputs [t, num_layers, PLE_DIM]: the scaled PLE
    // token embedding plus the normalized projection of the (scaled) token
    // embedding, averaged with a 1/sqrt(2) factor. The raw PLE rows either
    // come in pre-gathered from the host ([t, n*PLE_DIM]) or are gathered
    // here from the on-device table.
    fn per_layer_inputs(
        &self,
        tokens: &XlaOp,
        ple_rows: Option<&XlaOp>,
        x: &XlaOp,
        t: i64,
        dt: PrimitiveType,
    ) -> Result<XlaOp> {
        let b = tokens.builder();
        let n = self.num_layers as i64;
        let ple = match (ple_rows, &self.embed_per_layer) {
            (Some(rows), None) => rows.clone(),
            (None, Some(table)) => table.take(tokens, 0)?,
            _ => anyhow::bail!("exactly one of ple_rows and embed_per_layer must be set"),
        };
        let ple_scale = b.c0((PLE_DIM as f32).sqrt())?.convert(dt)?;
        let ple_scale = ple_scale.broadcast_in_dim(&[t, n * PLE_DIM], &[])?;
        let ple = ((ple * ple_scale)?).reshape(&[t, n, PLE_DIM])?;
        let proj = linear(x, &self.ple_projection)?;
        let proj_scale = b.c0(1f32 / (self.hidden_size as f32).sqrt())?.convert(dt)?;
        let proj_scale = proj_scale.broadcast_in_dim(&[t, n * PLE_DIM], &[])?;
        let proj = ((proj * proj_scale)?).reshape(&[t, n, PLE_DIM])?;
        let proj = rms_norm(&proj, &self.ple_projection_norm)?;
        let mix_scale = b.c0(0.5f32.sqrt())?.convert(dt)?;
        let mix_scale = mix_scale.broadcast_in_dim(&[t, n, PLE_DIM], &[])?;
        Ok(((proj + ple)? * mix_scale)?)
    }

    // Final norm, tied lm head and logit softcapping (in bf16, matching the
    // sequence of bf16-rounded ops of the reference implementation), returns
    // the argmax token.
    fn logits_argmax(&self, x: &XlaOp) -> Result<XlaOp> {
        let b = x.builder();
        let dt = x.ty()?;
        let x = rms_norm(x, &self.final_ln)?;
        let logits = linear(&x, &self.embed)?;
        let cap = b.c0(FINAL_LOGIT_SOFTCAP)?.convert(dt)?;
        let cap = cap.broadcast_in_dim(logits.array_shape()?.dims(), &[])?;
        let logits = ((logits / &cap)?.tanh()? * cap)?;
        Ok(logits.argmax(ElementType::S32, -1)?)
    }
}

// Rope tables as f32 constants, [T, hd]: cos/sin of position * inv_freq with
// the two half-dim copies concatenated. For the full attention layers only
// the first quarter of the angles has a non-zero frequency.
fn rope_tables(builder: &XlaBuilder, full_attention: bool) -> Result<(XlaOp, XlaOp)> {
    let (hd, theta, rotary_frac) = if full_attention {
        (GLOBAL_HEAD_DIM, GLOBAL_ROPE_THETA, GLOBAL_PARTIAL_ROTARY_FACTOR)
    } else {
        (HEAD_DIM, LOCAL_ROPE_THETA, 1.0)
    };
    let half = (hd / 2) as usize;
    let rope_angles = (rotary_frac * hd as f64 / 2.0) as usize;
    let mut inv_freqs = Vec::with_capacity(half);
    for i in 0..half {
        if i < rope_angles {
            inv_freqs.push(1f64 / theta.powf(2f64 * i as f64 / hd as f64));
        } else {
            inv_freqs.push(0f64);
        }
    }
    let mut cos_data = Vec::with_capacity(CONTEXT_SIZE * hd as usize);
    let mut sin_data = Vec::with_capacity(CONTEXT_SIZE * hd as usize);
    for t in 0..T {
        for rep in 0..2 {
            let _ = rep;
            for inv_freq in inv_freqs.iter() {
                let f = t as f64 * inv_freq;
                cos_data.push(f.cos() as f32);
                sin_data.push(f.sin() as f32);
            }
        }
    }
    let cos = builder.c1(&cos_data)?.reshape(&[T, hd])?;
    let sin = builder.c1(&sin_data)?.reshape(&[T, hd])?;
    Ok((cos, sin))
}

// Causal mask as a constant, [T, T]. Row i allows positions j <= i. The 512
// token sliding window is larger than the context so sliding attention uses
// the same mask.
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

// The prefill computation: full padded context (and its host-gathered PLE
// rows unless the table is on the device) in, next token plus the k/v caches
// of the non-shared layers out.
fn build_prefill(
    builder: &XlaBuilder,
    vb: &VarBuilder,
    cfg: &Config,
    ple_on_device: bool,
) -> Result<XlaComputation> {
    let tokens = builder.parameter(0, ElementType::S32, &[T], "tokens")?;
    let last_pos = builder.parameter(1, ElementType::S32, &[], "last_pos")?;
    let n = cfg.num_layers as i64;
    let ple_rows = if ple_on_device {
        None
    } else {
        Some(builder.parameter(2, vb.dtype(), &[T, n * PLE_DIM], "ple_rows")?)
    };
    let model = Model::new(vb, cfg, ple_on_device)?;
    let dt = vb.dtype().primitive_type();
    let (cos_l, sin_l) = rope_tables(builder, false)?;
    let (cos_g, sin_g) = rope_tables(builder, true)?;
    let mask = causal_mask(builder)?;

    let mut x = model.embed(&tokens, dt)?;
    let ple_inputs = model.per_layer_inputs(&tokens, ple_rows.as_ref(), &x, T, dt)?;
    let mut states = Vec::with_capacity(2 * cfg.first_kv_shared_layer);
    // Post-rope k and post-norm v of the donor layers, reused by the shared
    // layers of the same attention type.
    let mut donor_kv: Vec<Option<(XlaOp, XlaOp)>> = vec![None; cfg.num_layers];
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let (cos, sin) =
            if cfg.is_full_attention(layer_idx) { (&cos_g, &sin_g) } else { (&cos_l, &sin_l) };
        let x_norm = rms_norm(&x, &layer.input_ln)?;
        let q = layer.attn.q(&x_norm, cos, sin, T)?;
        let (k, v) = if cfg.is_kv_shared(layer_idx) {
            donor_kv[cfg.kv_donor(layer_idx)].clone().ok_or_else(|| anyhow!("missing donor kv"))?
        } else {
            let (k, v) = layer.attn.kv(&x_norm, cos, sin, T)?;
            states.push(k.clone());
            states.push(v.clone());
            donor_kv[layer_idx] = Some((k.clone(), v.clone()));
            (k, v)
        };
        let attn_out = layer.attn.attend(&q, &k, &v, &mask, T, T)?;
        let ple = ple_inputs.slice_in_dim1(layer_idx as i64, layer_idx as i64 + 1, 1)?;
        let ple = ple.reshape(&[T, PLE_DIM])?;
        x = layer.post_attention(&x, &attn_out, &ple, T)?;
    }

    let zero = builder.c0(0i32)?;
    let x_last = x.dynamic_slice(&[&last_pos, &zero], &[1, cfg.hidden_size])?;
    let next_token = model.logits_argmax(&x_last)?;

    let mut outputs = vec![next_token];
    outputs.extend(states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// The decode computation: a single token at a given position (and its
// host-gathered PLE row unless the table is on the device), plus the k/v
// caches in, next token plus the updated caches out. The caches are passed
// as parameters after the weights so that the weight parameter indices match
// the prefill computation.
fn build_decode(
    builder: &XlaBuilder,
    vb: &VarBuilder,
    cfg: &Config,
    ple_on_device: bool,
) -> Result<XlaComputation> {
    let token = builder.parameter(0, ElementType::S32, &[1], "token")?;
    let pos = builder.parameter(1, ElementType::S32, &[], "pos")?;
    let n = cfg.num_layers as i64;
    let ple_rows = if ple_on_device {
        None
    } else {
        Some(builder.parameter(2, vb.dtype(), &[1, n * PLE_DIM], "ple_rows")?)
    };
    let model = Model::new(vb, cfg, ple_on_device)?;
    let dt = vb.dtype().primitive_type();

    let mut param_idx = vb.next_index() as i64;
    let dtype = vb.dtype();
    let mut caches: Vec<Option<(XlaOp, XlaOp)>> = vec![None; cfg.num_layers];
    for (layer_idx, cache) in caches.iter_mut().enumerate().take(cfg.first_kv_shared_layer) {
        let dims = [T, cfg.num_kv_heads, cfg.head_dim(layer_idx)];
        let k =
            builder.parameter(param_idx, dtype, &dims, &format!("layers.{layer_idx}.k_cache"))?;
        let v = builder.parameter(
            param_idx + 1,
            dtype,
            &dims,
            &format!("layers.{layer_idx}.v_cache"),
        )?;
        param_idx += 2;
        *cache = Some((k, v));
    }

    let (cos_l, sin_l) = rope_tables(builder, false)?;
    let (cos_g, sin_g) = rope_tables(builder, true)?;
    let mask = causal_mask(builder)?;
    let zero = builder.c0(0i32)?;
    let mask = mask.dynamic_slice(&[&pos, &zero], &[1, T])?;

    let mut x = model.embed(&token, dt)?;
    let ple_inputs = model.per_layer_inputs(&token, ple_rows.as_ref(), &x, 1, dt)?;
    let mut new_states = Vec::with_capacity(2 * cfg.first_kv_shared_layer);
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let hd = cfg.head_dim(layer_idx);
        let (cos, sin) =
            if cfg.is_full_attention(layer_idx) { (&cos_g, &sin_g) } else { (&cos_l, &sin_l) };
        let cos = cos.dynamic_slice(&[&pos, &zero], &[1, hd])?;
        let sin = sin.dynamic_slice(&[&pos, &zero], &[1, hd])?;
        let x_norm = rms_norm(&x, &layer.input_ln)?;
        let q = layer.attn.q(&x_norm, &cos, &sin, 1)?;
        let (k_cache, v_cache) = if cfg.is_kv_shared(layer_idx) {
            caches[cfg.kv_donor(layer_idx)].clone().ok_or_else(|| anyhow!("missing donor cache"))?
        } else {
            let (k, v) = layer.attn.kv(&x_norm, &cos, &sin, 1)?;
            let (k_cache, v_cache) = caches[layer_idx].clone().unwrap();
            let k_cache = k_cache.dynamic_update_slice(&k, &[&pos, &zero, &zero])?;
            let v_cache = v_cache.dynamic_update_slice(&v, &[&pos, &zero, &zero])?;
            caches[layer_idx] = Some((k_cache.clone(), v_cache.clone()));
            new_states.push(k_cache.clone());
            new_states.push(v_cache.clone());
            (k_cache, v_cache)
        };
        let attn_out = layer.attn.attend(&q, &k_cache, &v_cache, &mask, 1, T)?;
        let ple = ple_inputs.slice_in_dim1(layer_idx as i64, layer_idx as i64 + 1, 1)?;
        let ple = ple.reshape(&[1, PLE_DIM])?;
        x = layer.post_attention(&x, &attn_out, &ple, 1)?;
    }

    let next_token = model.logits_argmax(&x)?;
    let mut outputs = vec![next_token];
    outputs.extend(new_states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// Download the tokenizer and weights from the hugging face hub, using the
// local cache if they have already been fetched. Note that the Gemma 4
// repositories are gated: the hugging face token from the standard locations
// (HF_TOKEN or ~/.cache/huggingface/token) is used, and the license has to be
// accepted on the model page first.
fn hub_model_files(repo: &str) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let client = hf_hub::HFClientSync::new()?;
    let repo = client.model("google", repo);
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

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
enum Which {
    #[value(name = "e2b")]
    E2b,
    #[value(name = "e4b")]
    E4b,
}

impl Which {
    fn config(self) -> &'static Config {
        match self {
            Self::E2b => &CONFIG_E2B,
            Self::E4b => &CONFIG_E4B,
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// The model size to run.
    #[arg(long, value_enum, default_value_t = Which::E2b)]
    which: Which,

    /// Keep the per-layer embedding table in device memory instead of
    /// gathering its rows on the host. Uses an extra 4.8GB (E2B) or 5.8GB
    /// (E4B, does not fit on a 16GB gpu) of device memory but restores the
    /// on-device chaining of the generated token in the decode loop.
    #[arg(long)]
    ple_on_device: bool,

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

fn make_client(force_cpu: bool) -> Result<PjRtClient> {
    if force_cpu {
        return Ok(PjRtClient::cpu()?);
    }
    match PjRtClient::gpu(0.90, false) {
        Ok(client) => return Ok(client),
        Err(err) => eprintln!("gpu client unavailable, falling back to cpu ({err})"),
    }
    Ok(PjRtClient::cpu()?)
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
    let cfg = args.which.config();
    let client = make_client(args.cpu)?;
    println!(
        "platform: {} {}, model: google/{}, dtype: bf16",
        client.platform_name(),
        client.platform_version(),
        cfg.repo,
    );

    let (tokenizer_path, weights_paths) = hub_model_files(cfg.repo)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("cannot load tokenizer: {e}"))?;
    let prompt = if args.raw_prompt {
        args.prompt.clone()
    } else {
        format!("<bos><|turn>user\n{}<turn|>\n<|turn>model\n", args.prompt)
    };
    let encoded = tokenizer.encode(prompt, false).map_err(|e| anyhow!("tokenizer error: {e}"))?;
    let mut tokens: Vec<i32> = encoded.get_ids().iter().map(|&t| t as i32).collect();
    println!("prompt has {} tokens", tokens.len());
    if tokens.is_empty() || tokens.len() >= CONTEXT_SIZE {
        anyhow::bail!("prompt length must be in [1, {}]", CONTEXT_SIZE - 1)
    }
    let stop_tokens: Vec<i32> = ["<turn|>", "<eos>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s).map(|t| t as i32))
        .collect();

    let start = std::time::Instant::now();
    // Non-weight args: token ids and position, plus the host-gathered PLE
    // rows unless the table lives on the device.
    let non_weight_args = if args.ple_on_device { 2 } else { 3 };
    let prefill_builder = XlaBuilder::new("gemma4-prefill");
    let vb = VarBuilder::new(&prefill_builder, ElementType::Bf16, non_weight_args);
    let prefill = build_prefill(&prefill_builder, &vb, cfg, args.ple_on_device)?;
    let decode_builder = XlaBuilder::new("gemma4-decode");
    let decode_vb = VarBuilder::new(&decode_builder, ElementType::Bf16, non_weight_args);
    let decode = build_decode(&decode_builder, &decode_vb, cfg, args.ple_on_device)?;
    println!("built the computations in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let (at_load, at_dump) = (at_load.as_deref(), at_dump.as_deref());
    let prefill_exe = client.compile_with_autotune_cache(&prefill, at_load, at_dump)?;
    let decode_exe = client.compile_with_autotune_cache(&decode, at_load, at_dump)?;
    println!("compiled the executables in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let weight_buffers = vb.load_buffers(&weights_paths, &client)?;
    let ple_table = if args.ple_on_device {
        None
    } else {
        Some(var_store::PleTable::load(
            &weights_paths,
            "model.language_model.embed_tokens_per_layer.weight",
            &[VOCAB_SIZE, cfg.num_layers as i64 * PLE_DIM],
            ElementType::Bf16,
        )?)
    };
    println!("loaded {} weights in {:?}", weight_buffers.len(), start.elapsed());

    let start = std::time::Instant::now();
    // Prefill: process the whole prompt, get the first token and the caches.
    let mut padded = tokens.clone();
    padded.resize(CONTEXT_SIZE, 0);
    let token_buffer = client.buffer_from_host_buffer(&padded, &[CONTEXT_SIZE], None)?;
    let pos_buffer = client.buffer_from_host_buffer(&[tokens.len() as i32 - 1], &[], None)?;
    let ple_buffer = match &ple_table {
        Some(table) => Some(table.gather_buffer(&client, &padded)?),
        None => None,
    };
    let mut inputs: Vec<&xla::PjRtBuffer> = vec![&token_buffer, &pos_buffer];
    inputs.extend(ple_buffer.iter());
    inputs.extend(weight_buffers.iter());
    let prefill_outputs = prefill_exe
        .execute_b(&inputs)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no execution result"))?;
    prefill_outputs[0].to_literal_sync()?;
    println!("prefill ({} tokens) in {:?}", tokens.len(), start.elapsed());
    let start = std::time::Instant::now();

    // Decode: one token at a time, the cache buffers stay on the device.
    let prompt_len = tokens.len();
    let mut generated = 0usize;
    let mut in_flight = prefill_outputs;
    match &ple_table {
        // PLE rows gathered on the host: the generated token has to come
        // back to the host before the next step can be launched.
        Some(table) => loop {
            // in_flight produces the token at position prompt_len + generated.
            let pos = prompt_len + generated;
            let next_token = in_flight[0].to_literal_sync()?.to_vec::<i32>()?[0];
            tokens.push(next_token);
            generated += 1;
            if generated >= args.sample_len
                || pos + 1 >= CONTEXT_SIZE
                || stop_tokens.contains(&next_token)
            {
                break;
            }
            let pos_buffer = client.buffer_from_host_buffer(&[pos as i32], &[], None)?;
            let ple_buffer = table.gather_buffer(&client, &[next_token])?;
            let mut inputs: Vec<&xla::PjRtBuffer> = vec![&in_flight[0], &pos_buffer, &ple_buffer];
            inputs.extend(weight_buffers.iter());
            inputs.extend(in_flight[1..].iter());
            in_flight = decode_exe
                .execute_b(&inputs)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no execution result"))?;
        },
        // PLE table on the device: the generated token is chained on the
        // device as in the qwen35 example, the next step is launched before
        // the token comes back to the host.
        None => loop {
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
        },
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
