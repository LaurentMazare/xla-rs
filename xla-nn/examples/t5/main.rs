// Inference example for the flan-T5 models, https://huggingface.co/google/flan-t5-large
//
// flan-T5 is an encoder-decoder model: a bidirectional encoder processes the
// whole prompt once, then the decoder generates tokens one at a time attending
// both to its own past (self attention) and to the encoder output (cross
// attention).
//
// The model files are downloaded automatically from the hugging face hub.
//
// Two computations are compiled:
// - The encoder one runs over the prompt and returns the final encoder hidden
//   states together with the per-layer cross-attention key/value projections of
//   these states, which is all the decoder ever needs from the encoder.
// - The decoder one processes a single position, using and updating the
//   per-layer self-attention kv cache. The cross-attention keys/values are
//   passed through unchanged.
// All the state tensors stay on the device across the generation steps. As the
// encoder length is only known once the prompt has been tokenized, the
// computations get built for the exact prompt length: nothing is padded on the
// encoder side and no encoder attention mask is needed.
//
// The implementation follows the reference one in transformers
// (models/t5/modeling_t5.py): the T5 layer norm is a rms norm with neither bias
// nor mean subtraction, the attention logits are *not* scaled by 1/sqrt(d_kv),
// the positional information comes from a learned relative attention bias
// shared across the layers of a stack, and the feed forward is a gated gelu
// using the tanh approximation.
use anyhow::{anyhow, Result};
use clap::Parser;

extern crate xla;
use xla::{ElementType, PjRtClient, PrimitiveType, XlaBuilder, XlaComputation, XlaOp};

use xla_nn::VarBuilder;

// Parameters 0 and 1 are reserved for the token ids and the position.
const NUM_NON_WEIGHT_ARGS: usize = 2;

// Configuration values shared by all the flan-T5 sizes.
const VOCAB_SIZE: i64 = 32128;
const D_KV: i64 = 64;
const REL_ATTN_NUM_BUCKETS: i64 = 32;
const REL_ATTN_MAX_DISTANCE: i64 = 128;
const LAYER_NORM_EPS: f32 = 1e-6;
// T5 starts the decoder on the padding token and stops on the eos token.
const DECODER_START_TOKEN: i32 = 0;
const EOS_TOKEN: i32 = 1;

#[derive(Clone, Copy, Debug)]
struct Config {
    repo: &'static str,
    d_model: i64,
    d_ff: i64,
    num_heads: i64,
    // The encoder and the decoder always have the same number of layers for the
    // flan-T5 checkpoints.
    num_layers: usize,
    num_shards: usize,
}

impl Config {
    // The attention inner dimension, which is not tied to d_model in T5.
    fn inner_dim(&self) -> i64 {
        self.num_heads * D_KV
    }
}

const CONFIG_SMALL: Config = Config {
    repo: "google/flan-t5-small",
    d_model: 512,
    d_ff: 1024,
    num_heads: 6,
    num_layers: 8,
    num_shards: 1,
};

const CONFIG_BASE: Config = Config {
    repo: "google/flan-t5-base",
    d_model: 768,
    d_ff: 2048,
    num_heads: 12,
    num_layers: 12,
    num_shards: 1,
};

const CONFIG_LARGE: Config = Config {
    repo: "google/flan-t5-large",
    d_model: 1024,
    d_ff: 2816,
    num_heads: 16,
    num_layers: 24,
    num_shards: 1,
};

const CONFIG_XL: Config = Config {
    repo: "google/flan-t5-xl",
    d_model: 2048,
    d_ff: 5120,
    num_heads: 32,
    num_layers: 24,
    num_shards: 2,
};

const CONFIG_XXL: Config = Config {
    repo: "google/flan-t5-xxl",
    d_model: 4096,
    d_ff: 10240,
    num_heads: 64,
    num_layers: 24,
    num_shards: 5,
};

fn linear(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    // x: [..., in], w: [out, in] -> [..., out]
    let x_rank = x.rank()? as i64;
    Ok(x.dot_general(w, &[x_rank - 1], &[1], &[], &[])?)
}

// T5LayerNorm: a rms norm with no bias and no mean subtraction. As in the
// reference implementation the sum of squares is accumulated in f32 and the
// normalized value is cast back before being scaled by the weight.
fn t5_layer_norm(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let variance = (&x * &x)?.reduce_mean(&[-1], true)?;
    let x_norm = (&x * (variance + b.c0(LAYER_NORM_EPS)?)?.rsqrt()?)?.convert(dt)?;
    let rank = x_norm.rank()? as i64;
    let w = w.broadcast_in_dim(x_norm.array_shape()?.dims(), &[rank - 1])?;
    Ok((w * x_norm)?)
}

// The `gelu_new` activation used by the gated feed forward, i.e. the tanh
// approximation of the gelu:
//   0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
fn gelu_new(x: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let c = |v: f32| -> Result<XlaOp> { Ok(b.c0(v)?.convert(dt)?) };
    let x3 = (x * (x * x)?)?;
    let inner = ((x + (x3 * c(0.044715)?)?)? * c((2f32 / std::f32::consts::PI).sqrt())?)?;
    Ok(((x * c(0.5)?)? * (inner.tanh()? + c(1f32)?)?)?)
}

// Port of T5Attention._relative_position_bucket for a single relative position.
// The reference computes the logarithmic part in f32, this does the same so that
// the bucket boundaries match exactly.
fn relative_position_bucket(relative_position: i64, bidirectional: bool) -> i64 {
    let mut num_buckets = REL_ATTN_NUM_BUCKETS;
    let mut relative_buckets = 0;
    let relative_position = if bidirectional {
        num_buckets /= 2;
        if relative_position > 0 {
            relative_buckets += num_buckets;
        }
        relative_position.abs()
    } else {
        -relative_position.min(0)
    };
    // Half of the buckets are for exact increments in positions, the other half
    // for logarithmically bigger bins up to max_distance.
    let max_exact = num_buckets / 2;
    if relative_position < max_exact {
        return relative_buckets + relative_position;
    }
    let ratio = (relative_position as f32 / max_exact as f32).ln();
    let scale = (REL_ATTN_MAX_DISTANCE as f64 / max_exact as f64).ln() as f32;
    let if_large = max_exact + (ratio / scale * (num_buckets - max_exact) as f32) as i64;
    relative_buckets + if_large.min(num_buckets - 1)
}

// The relative position bucket indices for a [q_len, k_len] attention, as a
// constant: they only depend on the positions. `memory_position - query_position`
// is the relative position, as in T5Attention.compute_bias.
fn bucket_ids(b: &XlaBuilder, q_len: i64, k_len: i64, bidirectional: bool) -> Result<XlaOp> {
    let mut ids = Vec::with_capacity((q_len * k_len) as usize);
    for q in 0..q_len {
        for k in 0..k_len {
            ids.push(relative_position_bucket(k - q, bidirectional) as i32)
        }
    }
    Ok(b.c1(&ids)?.reshape(&[q_len, k_len])?)
}

// Gather the per-head relative attention bias, w: [num_buckets, heads],
// ids: [q_len, k_len] -> [heads, q_len, k_len].
fn rel_attn_bias(w: &XlaOp, ids: &XlaOp) -> Result<XlaOp> {
    Ok(w.take(ids, 0)?.transpose(&[2, 0, 1])?)
}

// Additive causal mask as a constant, [t, t]. Row i allows positions j <= i.
fn causal_mask(b: &XlaBuilder, t: i64) -> Result<XlaOp> {
    let mut mask = vec![0f32; (t * t) as usize];
    for i in 0..t as usize {
        for j in 0..t as usize {
            if j > i {
                mask[i * t as usize + j] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(b.c1(&mask)?.reshape(&[t, t])?)
}

// The gated feed forward, T5DenseGatedActDense.
struct Ff {
    wi_0: XlaOp,
    wi_1: XlaOp,
    wo: XlaOp,
}

impl Ff {
    fn new(vb: &VarBuilder, p: &str, cfg: &Config) -> Result<Self> {
        let (h, i) = (cfg.d_model, cfg.d_ff);
        let wi_0 = vb.var(&format!("{p}.DenseReluDense.wi_0.weight"), &[i, h])?;
        let wi_1 = vb.var(&format!("{p}.DenseReluDense.wi_1.weight"), &[i, h])?;
        let wo = vb.var(&format!("{p}.DenseReluDense.wo.weight"), &[h, i])?;
        Ok(Self { wi_0, wi_1, wo })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let gated = (gelu_new(&linear(x, &self.wi_0)?)? * linear(x, &self.wi_1)?)?;
        linear(&gated, &self.wo)
    }
}

// A T5 attention block, used for the encoder self attention, the decoder self
// attention and the decoder cross attention. The relative attention bias is
// only held by the first layer of a stack and is passed in by the caller, hence
// it is not part of this struct.
struct Attention {
    q: XlaOp,
    k: XlaOp,
    v: XlaOp,
    o: XlaOp,
    num_heads: i64,
}

impl Attention {
    fn new(vb: &VarBuilder, p: &str, cfg: &Config) -> Result<Self> {
        let (h, i) = (cfg.d_model, cfg.inner_dim());
        let q = vb.var(&format!("{p}.q.weight"), &[i, h])?;
        let k = vb.var(&format!("{p}.k.weight"), &[i, h])?;
        let v = vb.var(&format!("{p}.v.weight"), &[i, h])?;
        let o = vb.var(&format!("{p}.o.weight"), &[h, i])?;
        Ok(Self { q, k, v, o, num_heads: cfg.num_heads })
    }

    // Project t positions with one of the q/k/v weights, [t, d_model] ->
    // [heads, t, d_kv].
    fn project(&self, w: &XlaOp, x: &XlaOp, t: i64) -> Result<XlaOp> {
        Ok(linear(x, w)?.reshape(&[t, self.num_heads, D_KV])?.swap_dims(0, 1)?)
    }

    fn query(&self, x: &XlaOp, t: i64) -> Result<XlaOp> {
        self.project(&self.q, x, t)
    }

    fn key_value(&self, x: &XlaOp, t: i64) -> Result<(XlaOp, XlaOp)> {
        Ok((self.project(&self.k, x, t)?, self.project(&self.v, x, t)?))
    }

    // Attend from q [heads, tq, d_kv] over k/v [heads, tk, d_kv] and apply the
    // output projection. T5 does not scale the logits, the whole scaling is
    // folded into the initialization. `bias` is the relative attention bias
    // [heads, tq, tk] and `mask` an additive mask [tq, tk], both optional. As in
    // the reference implementation the softmax accumulates in f32.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        q: &XlaOp,
        k: &XlaOp,
        v: &XlaOp,
        bias: Option<&XlaOp>,
        mask: Option<&XlaOp>,
        tq: i64,
        tk: i64,
    ) -> Result<XlaOp> {
        let dt = q.ty()?;
        let nh = self.num_heads;
        let mut scores = q.dot_general(k, &[2], &[2], &[0], &[0])?;
        if let Some(bias) = bias {
            scores = (scores + bias.convert(dt)?)?;
        }
        if let Some(mask) = mask {
            scores = (scores + mask.convert(dt)?.broadcast_in_dim(&[nh, tq, tk], &[1, 2])?)?;
        }
        let probs = scores.convert(PrimitiveType::F32)?.softmax(-1)?.convert(dt)?;
        let ctx = probs.dot_general(v, &[2], &[1], &[0], &[0])?;
        let ctx = ctx.swap_dims(0, 1)?.reshape(&[tq, nh * D_KV])?;
        linear(&ctx, &self.o)
    }
}

struct EncoderLayer {
    attn_ln: XlaOp,
    attn: Attention,
    ff_ln: XlaOp,
    ff: Ff,
}

impl EncoderLayer {
    fn new(vb: &VarBuilder, p: &str, cfg: &Config) -> Result<Self> {
        let attn_ln = vb.var(&format!("{p}.layer.0.layer_norm.weight"), &[cfg.d_model])?;
        let attn = Attention::new(vb, &format!("{p}.layer.0.SelfAttention"), cfg)?;
        let ff_ln = vb.var(&format!("{p}.layer.1.layer_norm.weight"), &[cfg.d_model])?;
        let ff = Ff::new(vb, &format!("{p}.layer.1"), cfg)?;
        Ok(Self { attn_ln, attn, ff_ln, ff })
    }

    // Bidirectional self attention over the s prompt positions followed by the
    // feed forward, both with a pre-norm and a residual connection.
    fn forward(&self, x: &XlaOp, bias: &XlaOp, s: i64) -> Result<XlaOp> {
        let x_norm = t5_layer_norm(x, &self.attn_ln)?;
        let q = self.attn.query(&x_norm, s)?;
        let (k, v) = self.attn.key_value(&x_norm, s)?;
        let x = (x + self.attn.attend(&q, &k, &v, Some(bias), None, s, s)?)?;
        let ff = self.ff.forward(&t5_layer_norm(&x, &self.ff_ln)?)?;
        Ok((x + ff)?)
    }
}

struct DecoderLayer {
    attn_ln: XlaOp,
    attn: Attention,
    cross_ln: XlaOp,
    cross_attn: Attention,
    ff_ln: XlaOp,
    ff: Ff,
}

impl DecoderLayer {
    fn new(vb: &VarBuilder, p: &str, cfg: &Config) -> Result<Self> {
        let attn_ln = vb.var(&format!("{p}.layer.0.layer_norm.weight"), &[cfg.d_model])?;
        let attn = Attention::new(vb, &format!("{p}.layer.0.SelfAttention"), cfg)?;
        let cross_ln = vb.var(&format!("{p}.layer.1.layer_norm.weight"), &[cfg.d_model])?;
        let cross_attn = Attention::new(vb, &format!("{p}.layer.1.EncDecAttention"), cfg)?;
        let ff_ln = vb.var(&format!("{p}.layer.2.layer_norm.weight"), &[cfg.d_model])?;
        let ff = Ff::new(vb, &format!("{p}.layer.2"), cfg)?;
        Ok(Self { attn_ln, attn, cross_ln, cross_attn, ff_ln, ff })
    }

    // A single decoder position: causal self attention over the kv cache, cross
    // attention over the encoder keys/values, then the feed forward. Returns the
    // layer output and the updated kv cache.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &XlaOp,
        pos: &XlaOp,
        bias: &XlaOp,
        mask: &XlaOp,
        k_cache: &XlaOp,
        v_cache: &XlaOp,
        cross_k: &XlaOp,
        cross_v: &XlaOp,
        t: i64,
        s: i64,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let b = x.builder();
        let zero = b.c0(0i32)?;

        let x_norm = t5_layer_norm(x, &self.attn_ln)?;
        let q = self.attn.query(&x_norm, 1)?;
        let (k, v) = self.attn.key_value(&x_norm, 1)?;
        let k_cache = k_cache.dynamic_update_slice(&k, &[&zero, pos, &zero])?;
        let v_cache = v_cache.dynamic_update_slice(&v, &[&zero, pos, &zero])?;
        let attn = self.attn.attend(&q, &k_cache, &v_cache, Some(bias), Some(mask), 1, t)?;
        let x = (x + attn)?;

        // The cross-attention keys and values are precomputed by the encoder
        // computation and there is nothing to mask as the encoder length is
        // exactly the prompt length. The cross attention carries no relative
        // position bias.
        let x_norm = t5_layer_norm(&x, &self.cross_ln)?;
        let q = self.cross_attn.query(&x_norm, 1)?;
        let cross = self.cross_attn.attend(&q, cross_k, cross_v, None, None, 1, s)?;
        let x = (x + cross)?;

        let ff = self.ff.forward(&t5_layer_norm(&x, &self.ff_ln)?)?;
        Ok(((x + ff)?, k_cache, v_cache))
    }
}

struct Model {
    shared: XlaOp,
    encoder: Vec<EncoderLayer>,
    encoder_rel_bias: XlaOp,
    encoder_final_ln: XlaOp,
    decoder: Vec<DecoderLayer>,
    decoder_rel_bias: XlaOp,
    decoder_final_ln: XlaOp,
    lm_head: XlaOp,
}

impl Model {
    // Both computations declare all the weights in this same order as they share
    // a single buffer list; each one only uses the subset it needs.
    fn new(vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let shared = vb.var("shared.weight", &[VOCAB_SIZE, cfg.d_model])?;
        // The relative attention bias is only learned by the first layer of each
        // stack and shared with all the other layers of that stack.
        let encoder_rel_bias = vb.var(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            &[REL_ATTN_NUM_BUCKETS, cfg.num_heads],
        )?;
        let mut encoder = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            encoder.push(EncoderLayer::new(vb, &format!("encoder.block.{layer_idx}"), cfg)?)
        }
        let encoder_final_ln = vb.var("encoder.final_layer_norm.weight", &[cfg.d_model])?;
        let decoder_rel_bias = vb.var(
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            &[REL_ATTN_NUM_BUCKETS, cfg.num_heads],
        )?;
        let mut decoder = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            decoder.push(DecoderLayer::new(vb, &format!("decoder.block.{layer_idx}"), cfg)?)
        }
        let decoder_final_ln = vb.var("decoder.final_layer_norm.weight", &[cfg.d_model])?;
        // flan-T5 does not tie the lm head to the embeddings, and consequently
        // does not scale the decoder output by 1/sqrt(d_model) either.
        let lm_head = vb.var("lm_head.weight", &[VOCAB_SIZE, cfg.d_model])?;
        Ok(Self {
            shared,
            encoder,
            encoder_rel_bias,
            encoder_final_ln,
            decoder,
            decoder_rel_bias,
            decoder_final_ln,
            lm_head,
        })
    }
}

// The encoder computation: the prompt tokens in, the final encoder hidden states
// plus the per-layer cross-attention keys/values out.
fn build_encoder(
    builder: &XlaBuilder,
    vb: &VarBuilder,
    cfg: &Config,
    s: i64,
) -> Result<XlaComputation> {
    let tokens = builder.parameter(0, ElementType::S32, &[s], "tokens")?;
    // Unused here, parameter 1 is the decoder position. It still has to be
    // declared so that the weights get the same parameter indices in both
    // computations.
    let _pos = builder.parameter(1, ElementType::S32, &[], "unused_pos")?;
    let model = Model::new(vb, cfg)?;
    let ids = bucket_ids(builder, s, s, true)?;
    let bias = rel_attn_bias(&model.encoder_rel_bias, &ids)?;

    let mut x = model.shared.take(&tokens, 0)?;
    for layer in model.encoder.iter() {
        x = layer.forward(&x, &bias, s)?;
    }
    let x = t5_layer_norm(&x, &model.encoder_final_ln)?;

    // The hidden states are only returned so that they can be compared with the
    // reference implementation, they are converted to f32 so that the dump does
    // not depend on the dtype the model runs in.
    let mut outputs = vec![x.convert(PrimitiveType::F32)?];
    for layer in model.decoder.iter() {
        let (k, v) = layer.cross_attn.key_value(&x, s)?;
        outputs.push(k);
        outputs.push(v);
    }
    Ok(builder.tuple(&outputs)?.build()?)
}

// The decoder computation: a single token at a given position plus the
// self-attention kv caches and the encoder cross-attention keys/values in, next
// token, logits and updated caches out. The state is passed as parameters after
// the weights so that the weight parameter indices match the encoder
// computation.
fn build_decoder(
    builder: &XlaBuilder,
    vb: &VarBuilder,
    cfg: &Config,
    s: i64,
    t: i64,
) -> Result<XlaComputation> {
    let token = builder.parameter(0, ElementType::S32, &[1], "token")?;
    let pos = builder.parameter(1, ElementType::S32, &[], "pos")?;
    let model = Model::new(vb, cfg)?;

    let mut param_idx = (NUM_NON_WEIGHT_ARGS + vb.num_vars()) as i64;
    let dtype = vb.dtype();
    let mut state_param = |name: String, dims: &[i64]| -> Result<XlaOp> {
        let op = builder.parameter(param_idx, dtype, dims, &name)?;
        param_idx += 1;
        Ok(op)
    };
    let mut caches = Vec::with_capacity(2 * cfg.num_layers);
    let nh = cfg.num_heads;
    for layer_idx in 0..cfg.num_layers {
        caches.push(state_param(format!("block.{layer_idx}.k_cache"), &[nh, t, D_KV])?);
        caches.push(state_param(format!("block.{layer_idx}.v_cache"), &[nh, t, D_KV])?);
    }
    let mut cross = Vec::with_capacity(2 * cfg.num_layers);
    for layer_idx in 0..cfg.num_layers {
        cross.push(state_param(format!("block.{layer_idx}.cross_k"), &[nh, s, D_KV])?);
        cross.push(state_param(format!("block.{layer_idx}.cross_v"), &[nh, s, D_KV])?);
    }

    // The relative attention bias and the causal mask for the single query
    // position that is being processed.
    let zero = builder.c0(0i32)?;
    let ids = bucket_ids(builder, t, t, false)?.dynamic_slice(&[&pos, &zero], &[1, t])?;
    let bias = rel_attn_bias(&model.decoder_rel_bias, &ids)?;
    let mask = causal_mask(builder, t)?.dynamic_slice(&[&pos, &zero], &[1, t])?;

    let mut x = model.shared.take(&token, 0)?;
    let mut new_caches = Vec::with_capacity(2 * cfg.num_layers);
    for (layer_idx, layer) in model.decoder.iter().enumerate() {
        let (k_cache, v_cache) = (&caches[2 * layer_idx], &caches[2 * layer_idx + 1]);
        let (cross_k, cross_v) = (&cross[2 * layer_idx], &cross[2 * layer_idx + 1]);
        let (y, k_cache, v_cache) =
            layer.forward(&x, &pos, &bias, &mask, k_cache, v_cache, cross_k, cross_v, t, s)?;
        x = y;
        new_caches.push(k_cache);
        new_caches.push(v_cache);
    }
    let x = t5_layer_norm(&x, &model.decoder_final_ln)?;
    let logits = linear(&x, &model.lm_head)?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?;

    let mut outputs = vec![next_token, logits];
    outputs.extend(new_caches);
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

// Write a f32 literal out as raw little-endian values. Only used to compare the
// intermediate tensors with the reference implementation, the shapes are known
// on the reading side.
fn dump_f32(literal: &xla::Literal, path: String) -> Result<()> {
    use std::io::Write;
    let values = literal.to_vec::<f32>()?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values.iter() {
        bytes.extend_from_slice(&v.to_le_bytes())
    }
    std::fs::File::create(path)?.write_all(&bytes)?;
    Ok(())
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
        let filename = if cfg.num_shards == 1 {
            "model.safetensors".to_string()
        } else {
            format!("model-{:05}-of-{:05}.safetensors", shard, cfg.num_shards)
        };
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
    #[value(name = "small")]
    Small,
    #[value(name = "base")]
    Base,
    #[value(name = "large")]
    Large,
    #[value(name = "xl")]
    Xl,
    #[value(name = "xxl")]
    Xxl,
}

impl Which {
    fn config(self) -> &'static Config {
        match self {
            Self::Small => &CONFIG_SMALL,
            Self::Base => &CONFIG_BASE,
            Self::Large => &CONFIG_LARGE,
            Self::Xl => &CONFIG_XL,
            Self::Xxl => &CONFIG_XXL,
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Run on cpu rather than on gpu.
    #[arg(long)]
    cpu: bool,

    /// The model size to run.
    #[arg(long, value_enum, default_value_t = Which::Large)]
    which: Which,

    /// The dtype used for the weights and most of the computation, the layer
    /// norms and the attention softmax always accumulate in f32. Note that the
    /// flan-T5 activations overflow in f16.
    #[arg(long, value_enum, default_value_t = Dtype::F32)]
    dtype: Dtype,

    /// The prompt fed to the encoder.
    #[arg(long, default_value = "Translate to German: The house is wonderful.")]
    prompt: String,

    /// The maximum number of tokens to generate, also the size the decoder kv
    /// cache is compiled for.
    #[arg(long, default_value_t = 64)]
    sample_len: usize,

    /// Ignore the eos token and always generate sample_len tokens, so that the
    /// decode rate is measured over a fixed number of steps.
    #[arg(long)]
    ignore_eos: bool,

    /// Number of times the encoder and the decode loop are run, the best
    /// timings being the ones reported. Only useful for benchmarking.
    #[arg(long, default_value_t = 1)]
    bench_reps: usize,

    /// When set, write the encoder output to <prefix>-encoder.bin and the logits
    /// of the first decoder step to <prefix>-logits.bin as raw f32 values, for
    /// comparison with the reference implementation.
    #[arg(long)]
    dump_prefix: Option<String>,

    /// Cache file for the gpu gemm autotuner results: loaded when the file
    /// exists, written otherwise. Pinning the autotuner results makes the
    /// performance reproducible and speeds up the compilation of later runs.
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
    let cfg = args.which.config();
    let client = PjRtClient::auto(args.cpu)?;
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
    // The T5 tokenizer appends the eos token to the prompt, which the reference
    // implementation relies on.
    let encoded = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| anyhow!("tokenizer error: {e}"))?;
    let prompt_tokens: Vec<i32> = encoded.get_ids().iter().map(|&t| t as i32).collect();
    println!("prompt has {} tokens: {prompt_tokens:?}", prompt_tokens.len());
    if prompt_tokens.is_empty() {
        anyhow::bail!("empty prompt")
    }
    if args.sample_len == 0 {
        anyhow::bail!("sample-len must be positive")
    }
    // The encoder computation is built for the exact prompt length, the decoder
    // one for the requested number of generation steps.
    let s = prompt_tokens.len() as i64;
    let t = args.sample_len as i64;

    let start = std::time::Instant::now();
    let encoder_builder = XlaBuilder::new("t5-encoder");
    let encoder_vb =
        VarBuilder::new(&encoder_builder, args.dtype.element_type(), NUM_NON_WEIGHT_ARGS);
    let encoder = build_encoder(&encoder_builder, &encoder_vb, cfg, s)?;
    let decoder_builder = XlaBuilder::new("t5-decoder");
    let decoder_vb =
        VarBuilder::new(&decoder_builder, args.dtype.element_type(), NUM_NON_WEIGHT_ARGS);
    let decoder = build_decoder(&decoder_builder, &decoder_vb, cfg, s, t)?;
    println!("built the computations in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let (at_load, at_dump) = (at_load.as_deref(), at_dump.as_deref());
    let encoder_exe = client.compile_with_autotune_cache(&encoder, at_load, at_dump)?;
    let decoder_exe = client.compile_with_autotune_cache(&decoder, at_load, at_dump)?;
    println!("compiled the executables in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let weight_buffers = encoder_vb.load_buffers(&weights_paths, &client)?;
    encoder_vb.check_all_used(&weights_paths)?;
    println!("loaded {} weights in {:?}", weight_buffers.len(), start.elapsed());

    // The encoder returns the final encoder hidden states followed by the
    // per-layer cross-attention keys and values, which are kept on the device
    // and fed to every decoder step.
    let token_buffer = client.buffer_from_host_buffer(&prompt_tokens, &[s as usize], None)?;
    let unused_pos = client.buffer_from_host_buffer(&[0i32], &[], None)?;
    let run_encoder = || -> Result<Vec<xla::PjRtBuffer>> {
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![&token_buffer, &unused_pos];
        inputs.extend(weight_buffers.iter());
        encoder_exe
            .execute_b(&inputs)?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("no execution result"))
    };

    let dims = [cfg.num_heads as usize, t as usize, D_KV as usize];
    let elem_size = match args.dtype {
        Dtype::F32 => 4,
        Dtype::Bf16 | Dtype::F16 => 2,
    };
    // The masked out cache entries still take part in the weighted sum with a
    // zero weight, so they have to hold actual zeros rather than garbage.
    let zeros = vec![0u8; dims.iter().product::<usize>() * elem_size];
    let mut init_caches: Vec<xla::PjRtBuffer> = Vec::with_capacity(2 * cfg.num_layers);
    for _ in 0..2 * cfg.num_layers {
        init_caches.push(client.buffer_from_host_raw_bytes(
            args.dtype.element_type(),
            &zeros,
            &dims,
            None,
        )?);
    }

    // Decode: one token at a time, the caches stay on the device.
    let decode = |encoder_outputs: &[xla::PjRtBuffer]| -> Result<(Vec<i32>, xla::Literal)> {
        let run_step = |token: &xla::PjRtBuffer,
                        pos: usize,
                        caches: &[xla::PjRtBuffer]|
         -> Result<Vec<xla::PjRtBuffer>> {
            let pos_buffer = client.buffer_from_host_buffer(&[pos as i32], &[], None)?;
            let mut inputs: Vec<&xla::PjRtBuffer> = vec![token, &pos_buffer];
            inputs.extend(weight_buffers.iter());
            inputs.extend(caches.iter());
            inputs.extend(encoder_outputs[1..].iter());
            decoder_exe
                .execute_b(&inputs)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no execution result"))
        };
        let start_token = client.buffer_from_host_buffer(&[DECODER_START_TOKEN], &[1], None)?;
        let mut in_flight = run_step(&start_token, 0, &init_caches)?;
        let first_logits = in_flight[1].to_literal_sync()?;
        let mut tokens: Vec<i32> = Vec::with_capacity(args.sample_len);
        loop {
            // in_flight holds the token generated at position tokens.len(),
            // which is also the token the next step consumes: it is fed back
            // straight from the device so that the next step gets dispatched
            // before the current one is read back, overlapping the host to
            // device round-trip and the dispatch with the device execution. The
            // step dispatched when the eos token shows up is wasted, but
            // harmless.
            let pos = tokens.len() + 1;
            let next = if pos < args.sample_len {
                Some(run_step(&in_flight[0], pos, &in_flight[2..])?)
            } else {
                None
            };
            let next_token = in_flight[0].to_literal_sync()?.to_vec::<i32>()?[0];
            tokens.push(next_token);
            match next {
                Some(o) if next_token != EOS_TOKEN || args.ignore_eos => in_flight = o,
                _ => break,
            }
        }
        Ok((tokens, first_logits))
    };

    // With more than one repetition the first one warms things up (the very
    // first execution of a compiled executable pays some one-off costs) and the
    // best timings are reported.
    let mut best_encoder = std::time::Duration::MAX;
    let mut best_decode = std::time::Duration::MAX;
    let (mut tokens, mut encoder_hidden, mut first_logits) = (Vec::new(), None, None);
    for _ in 0..args.bench_reps.max(1) {
        let start = std::time::Instant::now();
        let encoder_outputs = run_encoder()?;
        // Sync on the encoder output so that it can be timed separately from
        // the decode loop.
        let hidden = encoder_outputs[0].to_literal_sync()?;
        best_encoder = best_encoder.min(start.elapsed());
        let start = std::time::Instant::now();
        let (toks, logits) = decode(&encoder_outputs)?;
        best_decode = best_decode.min(start.elapsed());
        (tokens, encoder_hidden, first_logits) = (toks, Some(hidden), Some(logits));
    }
    println!("encoder ({s} tokens) in {best_encoder:?}");
    let tok_s = tokens.len() as f64 / best_decode.as_secs_f64();
    println!("decoded {} tokens in {best_decode:?} -> {tok_s:.1} tok/s", tokens.len());
    if let Some(prefix) = args.dump_prefix.as_ref() {
        dump_f32(encoder_hidden.as_ref().unwrap(), format!("{prefix}-encoder.bin"))?;
        dump_f32(first_logits.as_ref().unwrap(), format!("{prefix}-logits.bin"))?;
    }

    println!("generated ids: {tokens:?}");
    let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&ids, true).map_err(|e| anyhow!("tokenizer error: {e}"))?;
    println!("----\n{text}\n----");
    Ok(())
}
