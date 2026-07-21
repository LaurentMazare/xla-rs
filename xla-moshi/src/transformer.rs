//! The Mimi transformer, ported from `xn-moshi` and run over the whole sequence
//! at once. Only causal + norm-first + `kv_repeat = 1` + rope is supported,
//! which is what the Mimi configs use.
use crate::{Result, Vb};
use xla::{XlaBuilder, XlaOp};

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionalEmbedding {
    Rope,
    Sin,
    None,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub causal: bool,
    pub norm_first: bool,
    pub bias_ff: bool,
    pub bias_attn: bool,
    pub layer_scale: Option<f64>,
    pub positional_embedding: PositionalEmbedding,
    pub use_conv_block: bool,
    pub context: usize,
    pub max_period: f64,
    pub kv_repeat: usize,
    pub dim_feedforward: usize,
    pub conv_layout: bool,
}

struct Linear {
    weight: XlaOp,
    bias: Option<XlaOp>,
}

impl Linear {
    fn load(vb: &Vb, in_d: i64, out_d: i64, bias: bool) -> Result<Self> {
        let weight = vb.var("weight", &[out_d, in_d])?;
        let bias = if bias { Some(vb.var("bias", &[out_d])?) } else { None };
        Ok(Self { weight, bias })
    }

    fn from_weight(weight: XlaOp, bias: Option<XlaOp>) -> Self {
        Self { weight, bias }
    }

    fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let rank = xs.rank()? as i64;
        let ys = xs.dot_general(&self.weight, &[rank - 1], &[1], &[], &[])?;
        match &self.bias {
            None => Ok(ys),
            Some(b) => {
                let dims = ys.dims()?;
                let dims: Vec<i64> = dims.iter().map(|d| *d as i64).collect();
                let b = b.broadcast_in_dim(&dims, &[rank - 1])?;
                Ok(ys.add_(&b)?)
            }
        }
    }
}

struct LayerScale {
    scale: XlaOp,
}

impl LayerScale {
    fn load(vb: &Vb, d_model: i64) -> Result<Self> {
        Ok(Self { scale: vb.var("scale", &[d_model])? })
    }

    fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let d = self.scale.dims()?[0] as i64;
        let scale = self.scale.reshape(&[1, 1, d])?;
        Ok(xs.mul_(&scale)?)
    }
}

struct LayerNorm {
    weight: XlaOp,
    bias: XlaOp,
}

impl LayerNorm {
    fn load(vb: &Vb, d_model: i64) -> Result<Self> {
        Ok(Self { weight: vb.var("weight", &[d_model])?, bias: vb.var("bias", &[d_model])? })
    }

    fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        // xla's `layer_norm` multiplies/adds scale and bias without rank
        // broadcasting, so reshape them to `[1, 1, d]` to match `[b, t, d]`.
        let d = self.weight.dims()?[0] as i64;
        let weight = self.weight.reshape(&[1, 1, d])?;
        let bias = self.bias.reshape(&[1, 1, d])?;
        Ok(xs.layer_norm(-1, &weight, &bias)?)
    }
}

/// Precompute the interleaved-rope cos/sin tables for positions `0..t`, shaped
/// `[1, 1, t, head_dim / 2]` so they broadcast over batch and heads.
fn rope_tables(
    builder: &XlaBuilder,
    t: i64,
    head_dim: i64,
    max_period: f32,
) -> Result<(XlaOp, XlaOp)> {
    let half = (head_dim / 2) as usize;
    let mut cos = Vec::with_capacity(t as usize * half);
    let mut sin = Vec::with_capacity(t as usize * half);
    for pos in 0..t {
        for f in 0..half {
            let inv_freq = 1.0 / max_period.powf(f as f32 / half as f32);
            let freq = pos as f32 * inv_freq;
            cos.push(freq.cos());
            sin.push(freq.sin());
        }
    }
    let shape = [1, 1, t, half as i64];
    let cos = builder.constant_r1(&cos)?.reshape(&shape)?;
    let sin = builder.constant_r1(&sin)?.reshape(&shape)?;
    Ok((cos, sin))
}

/// Interleaved rope: rotates adjacent pairs `(x[2f], x[2f+1])`. `x` is
/// `[b, h, t, d]`, cos/sin are `[1, 1, t, d/2]`.
fn apply_rotary_emb(x: &XlaOp, cos: &XlaOp, sin: &XlaOp) -> Result<XlaOp> {
    let dims = x.dims()?;
    let (b, h, t, d) = (dims[0] as i64, dims[1] as i64, dims[2] as i64, dims[3] as i64);
    let half = d / 2;
    let x = x.reshape(&[b, h, t, half, 2])?;
    let x0 = x.slice_in_dim1(0, 1, 4)?.reshape(&[b, h, t, half])?;
    let x1 = x.slice_in_dim1(1, 2, 4)?.reshape(&[b, h, t, half])?;
    let cos = cos.broadcast_in_dim(&[b, h, t, half], &[0, 1, 2, 3])?;
    let sin = sin.broadcast_in_dim(&[b, h, t, half], &[0, 1, 2, 3])?;
    let o0 = x0.mul_(&cos)?.sub_(&x1.mul_(&sin)?)?;
    let o1 = x0.mul_(&sin)?.add_(&x1.mul_(&cos)?)?;
    let o0 = o0.reshape(&[b, h, t, half, 1])?;
    let o1 = o1.reshape(&[b, h, t, half, 1])?;
    Ok(o0.concat_in_dim(&[o1], 4)?.reshape(&[b, h, t, d])?)
}

/// Banded causal mask `[1, 1, t, t]`: query `i` attends to key `j` iff
/// `j <= i && i - j <= context` (matches `get_mask_abs` in the reference).
fn attn_mask(builder: &XlaBuilder, t: i64, context: i64) -> Result<XlaOp> {
    let mut m = Vec::with_capacity((t * t) as usize);
    for i in 0..t {
        for j in 0..t {
            let valid = j <= i && (i - j) <= context;
            m.push(if valid { 0f32 } else { f32::NEG_INFINITY });
        }
    }
    Ok(builder.constant_r1(&m)?.reshape(&[1, 1, t, t])?)
}

struct MultiheadAttention {
    in_proj: Linear,
    out_proj: Linear,
    num_heads: i64,
    head_dim: i64,
}

impl MultiheadAttention {
    fn load(vb: &Vb, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model as i64;
        let num_heads = cfg.num_heads as i64;
        let head_dim = d_model / num_heads;
        let num_kv = num_heads / cfg.kv_repeat as i64;
        let out_dim = d_model + 2 * num_kv * head_dim;
        let vb_attn = vb.pp("self_attn");
        let in_proj_weight = vb_attn.var("in_proj_weight", &[out_dim, d_model])?;
        let in_proj_bias =
            if cfg.bias_attn { Some(vb_attn.var("in_proj_bias", &[out_dim])?) } else { None };
        let in_proj = Linear::from_weight(in_proj_weight, in_proj_bias);
        let out_proj = Linear::load(&vb_attn.pp("out_proj"), d_model, d_model, cfg.bias_attn)?;
        Ok(Self { in_proj, out_proj, num_heads, head_dim })
    }

    fn forward(&self, xs: &XlaOp, cos: &XlaOp, sin: &XlaOp, mask: &XlaOp) -> Result<XlaOp> {
        let dims = xs.dims()?;
        let (b, t) = (dims[0] as i64, dims[1] as i64);
        let (h, hd) = (self.num_heads, self.head_dim);
        let d_model = h * hd;

        let qkv = self.in_proj.forward(xs)?;
        let split = |start: i64| -> Result<XlaOp> {
            let s = qkv.slice_in_dim1(start, start + d_model, 2)?;
            Ok(s.reshape(&[b, t, h, hd])?.transpose(&[0, 2, 1, 3])?)
        };
        let q = split(0)?;
        let k = split(d_model)?;
        let v = split(2 * d_model)?;

        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // attn = softmax(q @ k^T * scale + mask) @ v
        let scale = xs.builder().c0(1f32 / (hd as f32).sqrt())?;
        let attn = q.dot_general(&k, &[3], &[3], &[0, 1], &[0, 1])?; // [b, h, t, t]
        let attn = attn.mul_(&scale)?;
        let mask = mask.broadcast_in_dim(&[b, h, t, t], &[0, 1, 2, 3])?;
        let attn = attn.add_(&mask)?;
        let attn = attn.softmax(-1)?;
        let out = attn.dot_general(&v, &[3], &[2], &[0, 1], &[0, 1])?; // [b, h, t, hd]
        let out = out.transpose(&[0, 2, 1, 3])?.reshape(&[b, t, d_model])?;
        self.out_proj.forward(&out)
    }
}

struct Mlp {
    linear1: Linear,
    linear2: Linear,
}

impl Mlp {
    fn load(vb: &Vb, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model as i64;
        let ff = cfg.dim_feedforward as i64;
        let linear1 = Linear::load(&vb.pp("linear1"), d_model, ff, cfg.bias_ff)?;
        let linear2 = Linear::load(&vb.pp("linear2"), ff, d_model, cfg.bias_ff)?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let xs = self.linear1.forward(xs)?.gelu_erf()?;
        self.linear2.forward(&xs)
    }
}

struct TransformerLayer {
    self_attn: MultiheadAttention,
    mlp: Mlp,
    norm1: LayerNorm,
    norm2: LayerNorm,
    layer_scale_1: Option<LayerScale>,
    layer_scale_2: Option<LayerScale>,
}

impl TransformerLayer {
    fn load(vb: &Vb, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model as i64;
        let self_attn = MultiheadAttention::load(vb, cfg)?;
        let mlp = Mlp::load(vb, cfg)?;
        let norm1 = LayerNorm::load(&vb.pp("norm1"), d_model)?;
        let norm2 = LayerNorm::load(&vb.pp("norm2"), d_model)?;
        let (layer_scale_1, layer_scale_2) = if cfg.layer_scale.is_some() {
            (
                Some(LayerScale::load(&vb.pp("layer_scale_1"), d_model)?),
                Some(LayerScale::load(&vb.pp("layer_scale_2"), d_model)?),
            )
        } else {
            (None, None)
        };
        Ok(Self { self_attn, mlp, norm1, norm2, layer_scale_1, layer_scale_2 })
    }

    fn forward(&self, xs: &XlaOp, cos: &XlaOp, sin: &XlaOp, mask: &XlaOp) -> Result<XlaOp> {
        let norm1 = self.norm1.forward(xs)?;
        let mut attn = self.self_attn.forward(&norm1, cos, sin, mask)?;
        if let Some(ls) = &self.layer_scale_1 {
            attn = ls.forward(&attn)?;
        }
        let xs = xs.add_(&attn)?;
        let norm2 = self.norm2.forward(&xs)?;
        let mut mlp = self.mlp.forward(&norm2)?;
        if let Some(ls) = &self.layer_scale_2 {
            mlp = ls.forward(&mlp)?;
        }
        Ok(xs.add_(&mlp)?)
    }
}

struct Transformer {
    layers: Vec<TransformerLayer>,
    head_dim: i64,
    max_period: f32,
    context: i64,
}

impl Transformer {
    fn load(vb: &Vb, cfg: &Config) -> Result<Self> {
        if !cfg.causal || !cfg.norm_first || cfg.kv_repeat != 1 {
            return Err(err("only causal norm_first kv_repeat=1 transformers are supported"));
        }
        if cfg.positional_embedding != PositionalEmbedding::Rope {
            return Err(err("only rope positional embedding is supported"));
        }
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(TransformerLayer::load(&vb_layers.pp(i), cfg)?);
        }
        Ok(Self {
            layers,
            head_dim: (cfg.d_model / cfg.num_heads) as i64,
            max_period: cfg.max_period as f32,
            context: cfg.context as i64,
        })
    }

    /// `xs`: `[b, t, d_model]` -> `[b, t, d_model]`.
    fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let t = xs.dims()?[1] as i64;
        let builder = xs.builder();
        let (cos, sin) = rope_tables(builder, t, self.head_dim, self.max_period)?;
        let mask = attn_mask(builder, t, self.context)?;
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, &cos, &sin, &mask)?;
        }
        Ok(xs)
    }
}

pub struct ProjectedTransformer {
    input_proj: Option<Linear>,
    output_proj: Option<Linear>,
    transformer: Transformer,
    conv_layout: bool,
}

impl ProjectedTransformer {
    pub fn load(vb: &Vb, input_dim: i64, cfg: &Config) -> Result<Self> {
        let d_model = cfg.d_model as i64;
        let input_proj = if input_dim != d_model {
            Some(Linear::load(&vb.pp("input_proj"), input_dim, d_model, false)?)
        } else {
            None
        };
        let output_proj = if input_dim != d_model {
            Some(Linear::load(&vb.pp("output_proj").pp(0), d_model, input_dim, false)?)
        } else {
            None
        };
        let transformer = Transformer::load(&vb.pp("transformer"), cfg)?;
        Ok(Self { input_proj, output_proj, transformer, conv_layout: cfg.conv_layout })
    }

    /// `xs`: `[b, c, t]` (conv layout) -> `[b, c, t]`.
    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let xs = if self.conv_layout { xs.swap_dims(1, 2)? } else { xs.clone() };
        let xs = match &self.input_proj {
            Some(p) => p.forward(&xs)?,
            None => xs,
        };
        let xs = self.transformer.forward(&xs)?;
        let xs = match &self.output_proj {
            Some(p) => p.forward(&xs)?,
            None => xs,
        };
        if self.conv_layout {
            Ok(xs.swap_dims(1, 2)?)
        } else {
            Ok(xs)
        }
    }
}

fn err(msg: &str) -> crate::Error {
    crate::Error::Xla(xla::Error::XlaError { msg: msg.to_string(), backtrace: String::new() })
}
