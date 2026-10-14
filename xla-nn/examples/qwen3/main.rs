// Inference example for the Qwen3 dense models, https://huggingface.co/Qwen/Qwen3-4B
//
// Qwen3 is a standard decoder-only transformer: every layer uses full
// attention with GQA, a per-head rms-norm on the queries and keys before a
// full-width rotary embedding, and a SwiGLU MLP. The small sizes tie the
// output head to the token embeddings.
//
// The model files are downloaded automatically from the hugging face hub.
//
// Two computations are compiled for a fixed batch size and context size: a
// prefill one that processes the whole (padded) prompt for every batch row and
// returns the k/v caches together with the first generated token, and a decode
// one that processes a single position per row using and updating those
// caches. All state tensors stay on the device across the generation steps.
//
// With `--bench-steps`, the decode computation is also timed on its own: each
// step is dispatched and its token read back before the next one is launched
// (a synchronous step, the figure comparable to a streaming step of the TTS
// stack), and then with the next step dispatched before the token readback (the
// pipelined rate the generation loop reaches).
use anyhow::{anyhow, Result};
use clap::Parser;

extern crate xla;
use xla::{ElementType, PjRtClient, PrimitiveType, XlaBuilder, XlaComputation, XlaOp};

use xla_nn::VarBuilder;

// Parameters 0 and 1 are reserved for the token ids and the position.
const NUM_NON_WEIGHT_ARGS: usize = 2;

// Configuration shared by all the Qwen3 dense sizes.
const VOCAB_SIZE: i64 = 151936;
const RMS_NORM_EPS: f32 = 1e-6;
const HEAD_DIM: i64 = 128;
const ROPE_THETA: f64 = 1e6;

#[derive(Clone, Copy, Debug)]
struct Config {
    repo: &'static str,
    hidden_size: i64,
    intermediate_size: i64,
    num_layers: usize,
    num_heads: i64,
    num_kv_heads: i64,
    tie_embeddings: bool,
    num_shards: usize,
}

const CONFIG_0_6B: Config = Config {
    repo: "Qwen/Qwen3-0.6B",
    hidden_size: 1024,
    intermediate_size: 3072,
    num_layers: 28,
    num_heads: 16,
    num_kv_heads: 8,
    tie_embeddings: true,
    num_shards: 1,
};

const CONFIG_4B: Config = Config {
    repo: "Qwen/Qwen3-4B",
    hidden_size: 2560,
    intermediate_size: 9728,
    num_layers: 36,
    num_heads: 32,
    num_kv_heads: 8,
    tie_embeddings: true,
    num_shards: 3,
};

// The batch and context sizes the computations get compiled for.
#[derive(Clone, Copy, Debug)]
struct Shape {
    batch: i64,
    context: i64,
}

fn linear(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    // x: [..., in], w: [out, in] -> [..., out]
    let x_rank = x.rank()? as i64;
    Ok(x.dot_general(w, &[x_rank - 1], &[1], &[], &[])?)
}

// Plain rms-norm over the last dim, computed in f32 and cast back to the input
// dtype as in the reference implementation.
fn rms_norm(x: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let dt = x.ty()?;
    let x = x.convert(PrimitiveType::F32)?;
    let mean2 = (&x * &x)?.reduce_mean(&[-1], true)?;
    let x_norm = (&x * (mean2 + b.c0(RMS_NORM_EPS)?)?.rsqrt()?)?;
    let rank = x.rank()? as i64;
    let w = w.convert(PrimitiveType::F32)?.broadcast_in_dim(x.array_shape()?.dims(), &[rank - 1])?;
    Ok((x_norm * w)?.convert(dt)?)
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

// Full attention with GQA, qk-norm and a full-width rope.
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
        let q_proj = vb.var(&format!("{p}.q_proj.weight"), &[nh * HEAD_DIM, hidden])?;
        let k_proj = vb.var(&format!("{p}.k_proj.weight"), &[nkv * HEAD_DIM, hidden])?;
        let v_proj = vb.var(&format!("{p}.v_proj.weight"), &[nkv * HEAD_DIM, hidden])?;
        let o_proj = vb.var(&format!("{p}.o_proj.weight"), &[hidden, nh * HEAD_DIM])?;
        let q_norm = vb.var(&format!("{p}.q_norm.weight"), &[HEAD_DIM])?;
        let k_norm = vb.var(&format!("{p}.k_norm.weight"), &[HEAD_DIM])?;
        Ok(Self { q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, n_heads: nh, n_kv_heads: nkv })
    }

    // Rotate [b, t, h, HEAD_DIM] q or k values (rotate-half convention),
    // cos/sin: [t, HEAD_DIM].
    fn apply_rope(x: &XlaOp, cos: &XlaOp, sin: &XlaOp, b: i64, t: i64, h: i64) -> Result<XlaOp> {
        let x1 = x.slice_in_dim1(0, HEAD_DIM / 2, 3)?;
        let x2 = x.slice_in_dim1(HEAD_DIM / 2, HEAD_DIM, 3)?;
        let rotated = x2.neg()?.concat_in_dim(&[&x1], 3)?;
        let dt = x.ty()?;
        let cos_b = cos.convert(dt)?.broadcast_in_dim(&[b, t, h, HEAD_DIM], &[1, 3])?;
        let sin_b = sin.convert(dt)?.broadcast_in_dim(&[b, t, h, HEAD_DIM], &[1, 3])?;
        Ok(((x * cos_b)? + (rotated * sin_b)?)?)
    }

    // The projections for t positions of b rows: post-norm, post-rope
    // q [b, t, h, D] and k/v [b, t, kvh, D].
    fn qkv(
        &self,
        x: &XlaOp,
        cos: &XlaOp,
        sin: &XlaOp,
        b: i64,
        t: i64,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let (nh, nkv) = (self.n_heads, self.n_kv_heads);
        let q = linear(x, &self.q_proj)?.reshape(&[b, t, nh, HEAD_DIM])?;
        let q = rms_norm(&q, &self.q_norm)?;
        let k = linear(x, &self.k_proj)?.reshape(&[b, t, nkv, HEAD_DIM])?;
        let k = rms_norm(&k, &self.k_norm)?;
        let v = linear(x, &self.v_proj)?.reshape(&[b, t, nkv, HEAD_DIM])?;
        let q = Self::apply_rope(&q, cos, sin, b, t, nh)?;
        let k = Self::apply_rope(&k, cos, sin, b, t, nkv)?;
        Ok((q, k, v))
    }

    // Attend from q [b, t, h, D] over k/v [b, s, kvh, D] with mask [t, s]
    // (shared by the batch rows), then the output projection.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        q: &XlaOp,
        k: &XlaOp,
        v: &XlaOp,
        mask: &XlaOp,
        b: i64,
        t: i64,
        s: i64,
    ) -> Result<XlaOp> {
        let builder = q.builder();
        let (nh, nkv) = (self.n_heads, self.n_kv_heads);
        // [b, t, h, d] -> [b, h, t, d], and repeat the kv heads for GQA. The
        // repeat is a broadcast XLA folds into the attention fusions; folding
        // the query groups into the time dimension instead (a single
        // [b, kvh, g*t, d] x [b, kvh, s, d] contraction) measured 35% slower.
        let q = q.swap_dims(1, 2)?;
        let repeat_kv = |x: &XlaOp| -> Result<XlaOp> {
            let x = x.swap_dims(1, 2)?; // [b, kvh, s, d]
            let groups = nh / nkv;
            Ok(x.broadcast_in_dim(&[b, nkv, groups, s, HEAD_DIM], &[0, 1, 3, 4])?
                .reshape(&[b, nh, s, HEAD_DIM])?)
        };
        let k = repeat_kv(k)?;
        let v = repeat_kv(v)?;

        let dt = q.ty()?;
        let scale = builder.c0(1f32 / (HEAD_DIM as f32).sqrt())?.convert(dt)?;
        let scores = (q.dot_general(&k, &[3], &[3], &[0, 1], &[0, 1])? * scale)?; // [b, h, t, s]
        let mask_b = mask.convert(dt)?.broadcast_in_dim(&[b, nh, t, s], &[2, 3])?;
        // The softmax is computed in f32 as in the reference implementation.
        let probs = (scores + mask_b)?.convert(PrimitiveType::F32)?.softmax(-1)?.convert(dt)?;
        let ctx = probs.dot_general(&v, &[3], &[2], &[0, 1], &[0, 1])?; // [b, h, t, d]
        let ctx = ctx.swap_dims(1, 2)?.reshape(&[b, t, nh * HEAD_DIM])?;
        linear(&ctx, &self.o_proj)
    }

    // Full-context forward. Returns the outputs [b, T, HIDDEN] and the
    // post-rope k/v caches [b, T, kvh, D] (positions beyond the prompt hold
    // garbage and get overwritten by the decode steps).
    fn forward_prefill(
        &self,
        x: &XlaOp,
        cos: &XlaOp,
        sin: &XlaOp,
        mask: &XlaOp,
        sh: Shape,
        p: i64,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let (q, k, v) = self.qkv(x, cos, sin, sh.batch, p)?;
        let y = self.attend(&q, &k, &v, mask, sh.batch, p, p)?;
        // A prefill shorter than the context leaves the rest of the caches
        // zero; the decode steps overwrite those slots before attending them.
        let pad = |c: &XlaOp| -> Result<XlaOp> {
            if p == sh.context {
                return Ok(c.clone());
            }
            let zero = c.builder().c0(0f32)?.convert(c.ty()?)?;
            Ok(c.pad_in_dim(&zero, 1, 0, sh.context - p)?)
        };
        Ok((y, pad(&k)?, pad(&v)?))
    }

    // Single position forward at position pos for every row. cos/sin:
    // [1, HEAD_DIM] for that position, mask: [1, T] allowing positions <= pos.
    // Returns the output and the updated k/v caches.
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
        sh: Shape,
    ) -> Result<(XlaOp, XlaOp, XlaOp)> {
        let builder = x.builder();
        let (q, k, v) = self.qkv(x, cos, sin, sh.batch, 1)?;
        let zero = builder.c0(0i32)?;
        let k_cache = k_cache.dynamic_update_slice(&k, &[&zero, pos, &zero, &zero])?;
        let v_cache = v_cache.dynamic_update_slice(&v, &[&zero, pos, &zero, &zero])?;
        let y = self.attend(&q, &k_cache, &v_cache, mask, sh.batch, 1, sh.context)?;
        Ok((y, k_cache, v_cache))
    }
}

struct DecoderLayer {
    input_ln: XlaOp,
    post_ln: XlaOp,
    attn: Attention,
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
        let embed = vb.var("model.embed_tokens.weight", &[VOCAB_SIZE, hidden])?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let p = format!("model.layers.{layer_idx}");
            let input_ln = vb.var(&format!("{p}.input_layernorm.weight"), &[hidden])?;
            let post_ln = vb.var(&format!("{p}.post_attention_layernorm.weight"), &[hidden])?;
            let attn = Attention::new(vb, &format!("{p}.self_attn"), cfg)?;
            let mlp = Mlp::new(vb, &format!("{p}.mlp"), cfg)?;
            layers.push(DecoderLayer { input_ln, post_ln, attn, mlp });
        }
        let final_ln = vb.var("model.norm.weight", &[hidden])?;
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

    // The residual stack for x [b, t, HIDDEN]; `mix` runs the attention of a
    // layer on the normalized input.
    fn body(
        &self,
        x: XlaOp,
        mut mix: impl FnMut(usize, &DecoderLayer, &XlaOp) -> Result<XlaOp>,
    ) -> Result<XlaOp> {
        let mut x = x;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let residual = x.clone();
            let x_norm = rms_norm(&x, &layer.input_ln)?;
            let mixed = mix(layer_idx, layer, &x_norm)?;
            let x_mid = (residual + mixed)?;
            x = (&x_mid + layer.mlp.forward(&rms_norm(&x_mid, &layer.post_ln)?)?)?;
        }
        rms_norm(&x, &self.final_ln)
    }
}

// Rotary embedding tables as constants, [T, HEAD_DIM].
fn rope_tables(builder: &XlaBuilder, context: i64) -> Result<(XlaOp, XlaOp)> {
    let n = (context * HEAD_DIM) as usize;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);
    for t in 0..context {
        for i in 0..HEAD_DIM as usize {
            let j = i % (HEAD_DIM / 2) as usize;
            let inv_freq = 1f64 / ROPE_THETA.powf(2f64 * j as f64 / HEAD_DIM as f64);
            let f = t as f64 * inv_freq;
            cos_data.push(f.cos() as f32);
            sin_data.push(f.sin() as f32);
        }
    }
    let cos = builder.c1(&cos_data)?.reshape(&[context, HEAD_DIM])?;
    let sin = builder.c1(&sin_data)?.reshape(&[context, HEAD_DIM])?;
    Ok((cos, sin))
}

// Causal mask as a constant, [T, T]. Row i allows positions j <= i.
fn causal_mask(builder: &XlaBuilder, context: i64) -> Result<XlaOp> {
    let t = context as usize;
    let mut mask_data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            mask_data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Ok(builder.c1(&mask_data)?.reshape(&[context, context])?)
}

// The prefill computation: p padded prompt positions in for every row
// (tokens [b, p], the same last position for all rows), next tokens [b] plus
// all the per-layer caches (context length, zero beyond p) out.
fn build_prefill(
    builder: &XlaBuilder,
    vb: &VarBuilder,
    cfg: &Config,
    sh: Shape,
    p: i64,
) -> Result<XlaComputation> {
    let b = sh.batch;
    let tokens = builder.parameter(0, ElementType::S32, &[b, p], "tokens")?;
    let last_pos = builder.parameter(1, ElementType::S32, &[], "last_pos")?;
    let model = Model::new(vb, cfg)?;
    let (cos, sin) = rope_tables(builder, p)?;
    let mask = causal_mask(builder, p)?;

    let x = model.embed.take(&tokens.reshape(&[b * p])?, 0)?.reshape(&[b, p, cfg.hidden_size])?;
    let mut states = Vec::with_capacity(2 * cfg.num_layers);
    let x = model.body(x, |_, layer, x_norm| {
        let (y, k_cache, v_cache) = layer.attn.forward_prefill(x_norm, &cos, &sin, &mask, sh, p)?;
        states.push(k_cache);
        states.push(v_cache);
        Ok(y)
    })?;
    // Only the logits for the last position are needed.
    let zero = builder.c0(0i32)?;
    let x_last = x.dynamic_slice(&[&zero, &last_pos, &zero], &[b, 1, cfg.hidden_size])?;
    let x_last = x_last.reshape(&[b, cfg.hidden_size])?;
    let logits = linear(&x_last, model.lm_head())?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?; // [b]

    let mut outputs = vec![next_token];
    outputs.extend(states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// The decode computation: one token per row at a given position plus the
// per-layer caches in, next tokens plus the updated caches out. The caches are
// passed as parameters after the weights so that the weight parameter indices
// match the prefill computation.
fn build_decode(
    builder: &XlaBuilder,
    vb: &VarBuilder,
    cfg: &Config,
    sh: Shape,
) -> Result<XlaComputation> {
    let (b, t) = (sh.batch, sh.context);
    let token = builder.parameter(0, ElementType::S32, &[b], "token")?;
    let pos = builder.parameter(1, ElementType::S32, &[], "pos")?;
    let model = Model::new(vb, cfg)?;

    let mut param_idx = (NUM_NON_WEIGHT_ARGS + vb.num_vars()) as i64;
    let dtype = vb.dtype();
    let mut states = Vec::with_capacity(2 * cfg.num_layers);
    for layer_idx in 0..cfg.num_layers {
        let dims = [b, t, cfg.num_kv_heads, HEAD_DIM];
        for name in ["k_cache", "v_cache"] {
            let p = builder.parameter(param_idx, dtype, &dims, &format!("layers.{layer_idx}.{name}"))?;
            param_idx += 1;
            states.push(p);
        }
    }

    let (cos, sin) = rope_tables(builder, t)?;
    let mask = causal_mask(builder, t)?;
    let zero = builder.c0(0i32)?;
    let cos = cos.dynamic_slice(&[&pos, &zero], &[1, HEAD_DIM])?;
    let sin = sin.dynamic_slice(&[&pos, &zero], &[1, HEAD_DIM])?;
    let mask = mask.dynamic_slice(&[&pos, &zero], &[1, t])?;

    let x = model.embed.take(&token, 0)?.reshape(&[b, 1, cfg.hidden_size])?;
    let mut new_states = Vec::with_capacity(2 * cfg.num_layers);
    let x = model.body(x, |layer_idx, layer, x_norm| {
        let (k_cache, v_cache) = (&states[2 * layer_idx], &states[2 * layer_idx + 1]);
        let (y, k_cache, v_cache) =
            layer.attn.forward_step(x_norm, &pos, &cos, &sin, &mask, k_cache, v_cache, sh)?;
        new_states.push(k_cache);
        new_states.push(v_cache);
        Ok(y)
    })?;
    let x = x.reshape(&[b, cfg.hidden_size])?;
    let logits = linear(&x, model.lm_head())?.convert(PrimitiveType::F32)?;
    let next_token = logits.argmax(ElementType::S32, -1)?; // [b]

    let mut outputs = vec![next_token];
    outputs.extend(new_states);
    Ok(builder.tuple(&outputs)?.build()?)
}

// Download the tokenizer and weight shards from the hugging face hub, using
// the local cache if they have already been fetched.
fn hub_model_files(cfg: &Config) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let client = hf_hub::HFClientSync::new()?;
    let (owner, name) = cfg.repo.split_once('/').ok_or_else(|| anyhow!("invalid repo"))?;
    let repo = client.model(owner, name);
    let tokenizer = repo.download_file().filename("tokenizer.json").send()?;
    let mut weights = Vec::with_capacity(cfg.num_shards);
    for shard in 1..=cfg.num_shards {
        let filename = if cfg.num_shards == 1 {
            "model.safetensors".to_string()
        } else {
            format!("model-{:05}-of-{:05}.safetensors", shard, cfg.num_shards)
        };
        weights.push(repo.download_file().filename(&filename).send()?);
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
    #[value(name = "0.6b")]
    W0_6b,
    #[value(name = "4b")]
    W4b,
}

impl Which {
    fn config(self) -> &'static Config {
        match self {
            Self::W0_6b => &CONFIG_0_6B,
            Self::W4b => &CONFIG_4B,
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Run on cpu rather than on gpu.
    #[arg(long)]
    cpu: bool,

    /// The model size to run.
    #[arg(long, value_enum, default_value_t = Which::W4b)]
    which: Which,

    /// The dtype used for the weights and most of the computation; the norms
    /// and the attention softmax always run in f32.
    #[arg(long, value_enum, default_value_t = Dtype::Bf16)]
    dtype: Dtype,

    /// The number of rows generated in parallel (the same prompt for all).
    #[arg(long, default_value_t = 1)]
    batch_size: usize,

    /// The context size the computations are compiled for (kv-cache length).
    #[arg(long, default_value_t = 512)]
    context: usize,

    /// The number of prompt positions the prefill computation processes
    /// (defaults to the context); the prompt is padded to it.
    #[arg(long)]
    prefill_len: Option<usize>,

    /// The prompt used for generation.
    #[arg(long, default_value = "What is the capital of France? Answer in one word.")]
    prompt: String,

    /// The maximum number of tokens to generate.
    #[arg(long, default_value_t = 30)]
    sample_len: usize,

    /// Feed the raw prompt to the model rather than using the chat template.
    #[arg(long)]
    raw_prompt: bool,

    /// Also time the decode step on its own for this many steps (after 10
    /// untimed ones), synchronous and pipelined, and print the statistics.
    #[arg(long, default_value_t = 0)]
    bench_steps: usize,

    /// Cache file for the gpu gemm autotuner results (see the qwen35 example).
    #[arg(long)]
    autotune_cache: Option<std::path::PathBuf>,
}

fn stats(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    (q(0.5), q(0.9), v.iter().sum::<f64>() / v.len() as f64)
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
    let sh = Shape { batch: args.batch_size as i64, context: args.context as i64 };
    let client = PjRtClient::auto(args.cpu)?;
    println!(
        "platform: {} {}, model: {}, dtype: {:?}, batch {}, context {}, prefill {}",
        client.platform_name(),
        client.platform_version(),
        cfg.repo,
        args.dtype,
        sh.batch,
        sh.context,
        args.prefill_len.unwrap_or(args.context)
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
    let prefill_len = args.prefill_len.unwrap_or(args.context);
    if tokens.is_empty() || tokens.len() > prefill_len || prefill_len > args.context {
        anyhow::bail!("prompt length must be in [1, {prefill_len}], prefill_len <= context")
    }
    let stop_tokens: Vec<i32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s).map(|t| t as i32))
        .collect();

    let start = std::time::Instant::now();
    let prefill_builder = XlaBuilder::new("qwen3-prefill");
    let vb = VarBuilder::new(&prefill_builder, args.dtype.element_type(), NUM_NON_WEIGHT_ARGS);
    let prefill = build_prefill(&prefill_builder, &vb, cfg, sh, prefill_len as i64)?;
    let decode_builder = XlaBuilder::new("qwen3-decode");
    let decode_vb =
        VarBuilder::new(&decode_builder, args.dtype.element_type(), NUM_NON_WEIGHT_ARGS);
    let decode = build_decode(&decode_builder, &decode_vb, cfg, sh)?;
    println!("built the computations in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let (at_load, at_dump) = (at_load.as_deref(), at_dump.as_deref());
    let prefill_exe = client.compile_with_autotune_cache(&prefill, at_load, at_dump)?;
    let decode_exe = client.compile_with_autotune_cache(&decode, at_load, at_dump)?;
    println!("compiled the executables in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let weight_buffers = vb.load_buffers(&weights_paths, &client)?;
    // The small checkpoints ship an lm_head.weight tensor even though the
    // head is tied to the embeddings.
    vb.check_all_used_with_ignore(&weights_paths, |name| {
        cfg.tie_embeddings && name == "lm_head.weight"
    })?;
    println!("loaded {} weights in {:?}", weight_buffers.len(), start.elapsed());

    let b = args.batch_size;
    let start = std::time::Instant::now();
    // Prefill: every row gets the same padded prompt.
    let mut padded = tokens.clone();
    padded.resize(prefill_len, 0);
    let batched: Vec<i32> = padded.iter().cycle().take(b * prefill_len).copied().collect();
    let token_buffer = client.buffer_from_host_buffer(&batched, &[b, prefill_len], None)?;
    let pos_buffer = client.buffer_from_host_buffer(&[tokens.len() as i32 - 1], &[], None)?;
    let mut inputs: Vec<&xla::PjRtBuffer> = vec![&token_buffer, &pos_buffer];
    inputs.extend(weight_buffers.iter());
    let prefill_outputs = prefill_exe
        .execute_b(&inputs)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no execution result"))?;
    prefill_outputs[0].to_literal_sync()?;
    println!("prefill ({} tokens x {b} rows) in {:?}", tokens.len(), start.elapsed());

    let run_step = |token: &xla::PjRtBuffer,
                    pos: usize,
                    states: &[xla::PjRtBuffer]|
     -> Result<Vec<xla::PjRtBuffer>> {
        let pos_buffer = client.buffer_from_host_buffer(&[pos as i32], &[], None)?;
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![token, &pos_buffer];
        inputs.extend(weight_buffers.iter());
        inputs.extend(states.iter());
        decode_exe.execute_b(&inputs)?.into_iter().next().ok_or_else(|| anyhow!("no result"))
    };

    // Decode: one token per row at a time, the state buffers stay on the
    // device. The generated token is chained on the device: the next step is
    // dispatched before the current token is read back, so that the dispatch
    // overlaps with the device execution. Row 0 drives the stop condition.
    let start = std::time::Instant::now();
    let prompt_len = tokens.len();
    let mut generated = 0usize;
    let mut in_flight = prefill_outputs;
    loop {
        let pos = prompt_len + generated;
        let next_outputs = if generated + 1 < args.sample_len && pos + 1 < args.context {
            Some(run_step(&in_flight[0], pos, &in_flight[1..])?)
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
        println!(
            "decoded {decode_steps} steps in {dt:?} -> {tok_s:.1} steps/s, {:.1} tok/s over {b} rows",
            tok_s * b as f64
        );
    }
    let all_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&all_ids, false).map_err(|e| anyhow!("tokenizer error: {e}"))?;
    println!("----\n{text}\n----");

    if args.bench_steps > 0 {
        // Synchronous steps: dispatch, read the tokens back, repeat. The
        // position stays in the middle of the context so every step attends
        // over the same number of cache slots.
        let pos = args.context / 2;
        let mut states = in_flight;
        let mut sync_ms = vec![];
        for i in 0..10 + args.bench_steps {
            let t0 = std::time::Instant::now();
            let mut out = run_step(&states[0], pos, &states[1..])?;
            let token = out.remove(0);
            token.to_literal_sync()?;
            states = std::iter::once(token).chain(out).collect();
            if i >= 10 {
                sync_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let (p50, p90, mean) = stats(&mut sync_ms);
        println!("bench,sync,{b},{},{p50:.3},{p90:.3},{mean:.3}", args.context);
        // Pipelined: the next step is dispatched before the readback of the
        // current one, as in the generation loop.
        let t0 = std::time::Instant::now();
        let mut in_flight = states;
        for _ in 0..args.bench_steps {
            let next = run_step(&in_flight[0], pos, &in_flight[1..])?;
            in_flight[0].to_literal_sync()?;
            in_flight = next;
        }
        in_flight[0].to_literal_sync()?;
        let per_step = t0.elapsed().as_secs_f64() * 1000.0 / args.bench_steps as f64;
        println!("bench,pipelined,{b},{},{per_step:.3},,{per_step:.3}", args.context);
        // Prefill: the prompt positions for every row, synced on the first
        // generated token (the outputs are dropped, the decode state above is
        // not touched).
        let mut pre_ms = vec![];
        for i in 0..10 + args.bench_steps {
            let t0 = std::time::Instant::now();
            let mut inputs: Vec<&xla::PjRtBuffer> = vec![&token_buffer, &pos_buffer];
            inputs.extend(weight_buffers.iter());
            let out = prefill_exe
                .execute_b(&inputs)?
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no execution result"))?;
            out[0].to_literal_sync()?;
            if i >= 10 {
                pre_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let (p50, p90, mean) = stats(&mut pre_ms);
        println!("bench,prefill,{b},{prefill_len},{p50:.3},{p90:.3},{mean:.3}");
    }
    Ok(())
}
