//! SEANet encoder/decoder, ported from `xn-moshi` (forward path only).
use crate::conv::{Norm, PadMode, StreamableConv1d, StreamableConvTranspose1d};
use crate::{Result, StepCtx, Vb};
use xla::XlaOp;

#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Elu(f32),
    Gelu,
    Relu,
    Silu,
    Tanh,
    Sigmoid,
}

impl Activation {
    pub fn apply(&self, xs: &XlaOp) -> Result<XlaOp> {
        let b = xs.builder();
        let zero = b.c0(0f32)?;
        match self {
            // elu(x) = max(x, 0) + min(alpha * (exp(x) - 1), 0)
            Activation::Elu(alpha) => {
                let pos = xs.max(&zero)?;
                let neg = xs.expm1()?.mul_(&b.c0(*alpha)?)?.min(&zero)?;
                Ok(pos.add_(&neg)?)
            }
            Activation::Gelu => Ok(xs.gelu_erf()?),
            Activation::Relu => Ok(xs.max(&zero)?),
            Activation::Silu => Ok(xs.silu()?),
            Activation::Tanh => Ok(xs.tanh()?),
            Activation::Sigmoid => Ok(xs.sigmoid()?),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub dimension: usize,
    pub channels: usize,
    pub causal: bool,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub activation: Activation,
    pub norm: Norm,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: PadMode,
    pub true_skip: bool,
    pub compress: usize,
    pub lstm: Option<usize>,
    pub disable_norm_outer_blocks: usize,
    pub final_activation: Option<Activation>,
}

struct SeaNetResnetBlock {
    block: Vec<StreamableConv1d>,
    shortcut: Option<StreamableConv1d>,
    activation: Activation,
}

impl SeaNetResnetBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        vb: &Vb,
        dim: i64,
        k_sizes_and_dilations: &[(i64, i64)],
        activation: Activation,
        norm: Option<Norm>,
        causal: bool,
        pad_mode: PadMode,
        compress: i64,
        true_skip: bool,
    ) -> Result<Self> {
        let hidden = dim / compress;
        let vb_b = vb.pp("block");
        let n = k_sizes_and_dilations.len();
        let mut block = Vec::with_capacity(n);
        for (i, &(k_size, dilation)) in k_sizes_and_dilations.iter().enumerate() {
            let in_c = if i == 0 { dim } else { hidden };
            let out_c = if i == n - 1 { dim } else { hidden };
            block.push(StreamableConv1d::load(
                &vb_b.pp(2 * i as i64 + 1),
                in_c,
                out_c,
                k_size,
                1,
                dilation,
                1,
                true,
                causal,
                norm,
                pad_mode,
            )?);
        }
        let shortcut = if true_skip {
            None
        } else {
            Some(StreamableConv1d::load(
                &vb.pp("shortcut"),
                dim,
                dim,
                1,
                1,
                1,
                1,
                true,
                causal,
                norm,
                pad_mode,
            )?)
        };
        Ok(Self { block, shortcut, activation })
    }

    fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let mut ys = xs.clone();
        for conv in &self.block {
            ys = self.activation.apply(&ys)?;
            ys = conv.forward(&ys)?;
        }
        match &self.shortcut {
            None => Ok(ys.add_(xs)?),
            Some(shortcut) => Ok(ys.add_(&shortcut.forward(xs)?)?),
        }
    }

    fn step(&self, xs: &XlaOp, ctx: &mut StepCtx) -> Result<XlaOp> {
        let mut ys = xs.clone();
        for conv in &self.block {
            ys = self.activation.apply(&ys)?;
            ys = conv.step(&ys, ctx)?;
        }
        match &self.shortcut {
            None => Ok(ys.add_(xs)?),
            Some(shortcut) => Ok(ys.add_(&shortcut.step(xs, ctx)?)?),
        }
    }
}

struct EncoderLayer {
    residuals: Vec<SeaNetResnetBlock>,
    downsample: StreamableConv1d,
}

pub struct SeaNetEncoder {
    init_conv: StreamableConv1d,
    activation: Activation,
    layers: Vec<EncoderLayer>,
    final_conv: StreamableConv1d,
}

impl SeaNetEncoder {
    pub fn load(vb: &Vb, cfg: &Config) -> Result<Self> {
        if cfg.lstm.unwrap_or(0) > 0 {
            return Err(err("seanet lstm is not supported"));
        }
        let n_blocks = 2 + cfg.ratios.len();
        let nf = cfg.n_filters as i64;
        let mut mult = 1i64;
        let init_norm = if cfg.disable_norm_outer_blocks >= 1 { None } else { Some(cfg.norm) };
        let mut layer_idx = 0i64;
        let vb = vb.pp("model");
        let init_conv = StreamableConv1d::load(
            &vb.pp(layer_idx),
            cfg.channels as i64,
            mult * nf,
            cfg.kernel_size as i64,
            1,
            1,
            1,
            true,
            cfg.causal,
            init_norm,
            cfg.pad_mode,
        )?;
        layer_idx += 1;

        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for (i, &ratio) in cfg.ratios.iter().rev().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks >= i + 2 { None } else { Some(cfg.norm) };
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let dilation = (cfg.dilation_base as i64).pow(j as u32);
                residuals.push(SeaNetResnetBlock::load(
                    &vb.pp(layer_idx),
                    mult * nf,
                    &[(cfg.residual_kernel_size as i64, dilation), (1, 1)],
                    cfg.activation,
                    norm,
                    cfg.causal,
                    cfg.pad_mode,
                    cfg.compress as i64,
                    cfg.true_skip,
                )?);
                layer_idx += 1;
            }
            let downsample = StreamableConv1d::load(
                &vb.pp(layer_idx + 1),
                mult * nf,
                mult * nf * 2,
                ratio as i64 * 2,
                ratio as i64,
                1,
                1,
                true,
                true,
                norm,
                cfg.pad_mode,
            )?;
            layer_idx += 2;
            layers.push(EncoderLayer { residuals, downsample });
            mult *= 2;
        }

        let final_norm =
            if cfg.disable_norm_outer_blocks >= n_blocks { None } else { Some(cfg.norm) };
        let final_conv = StreamableConv1d::load(
            &vb.pp(layer_idx + 1),
            mult * nf,
            cfg.dimension as i64,
            cfg.last_kernel_size as i64,
            1,
            1,
            1,
            true,
            cfg.causal,
            final_norm,
            cfg.pad_mode,
        )?;

        Ok(Self { init_conv, activation: cfg.activation, layers, final_conv })
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let mut xs = self.init_conv.forward(xs)?;
        for layer in &self.layers {
            for residual in &layer.residuals {
                xs = residual.forward(&xs)?;
            }
            xs = self.activation.apply(&xs)?;
            xs = layer.downsample.forward(&xs)?;
        }
        xs = self.activation.apply(&xs)?;
        self.final_conv.forward(&xs)
    }

    /// Streaming step: `xs` is `[1, channels, l]`, returns `[1, dimension, l / 960]`
    /// (the SEANet downsampling factor) worth of encoder frames.
    pub fn step(&self, xs: &XlaOp, ctx: &mut StepCtx) -> Result<XlaOp> {
        let mut xs = self.init_conv.step(xs, ctx)?;
        for layer in &self.layers {
            for residual in &layer.residuals {
                xs = residual.step(&xs, ctx)?;
            }
            xs = self.activation.apply(&xs)?;
            xs = layer.downsample.step(&xs, ctx)?;
        }
        xs = self.activation.apply(&xs)?;
        self.final_conv.step(&xs, ctx)
    }
}

struct DecoderLayer {
    upsample: StreamableConvTranspose1d,
    residuals: Vec<SeaNetResnetBlock>,
}

pub struct SeaNetDecoder {
    init_conv: StreamableConv1d,
    activation: Activation,
    layers: Vec<DecoderLayer>,
    final_conv: StreamableConv1d,
    final_activation: Option<Activation>,
}

impl SeaNetDecoder {
    pub fn load(vb: &Vb, cfg: &Config) -> Result<Self> {
        if cfg.lstm.unwrap_or(0) > 0 {
            return Err(err("seanet lstm is not supported"));
        }
        let n_blocks = 2 + cfg.ratios.len();
        let nf = cfg.n_filters as i64;
        let mut mult = 1i64 << cfg.ratios.len();
        let init_norm =
            if cfg.disable_norm_outer_blocks == n_blocks { None } else { Some(cfg.norm) };
        let mut layer_idx = 0i64;
        let vb = vb.pp("model");
        let init_conv = StreamableConv1d::load(
            &vb.pp(layer_idx),
            cfg.dimension as i64,
            mult * nf,
            cfg.kernel_size as i64,
            1,
            1,
            1,
            true,
            cfg.causal,
            init_norm,
            cfg.pad_mode,
        )?;
        layer_idx += 1;

        let mut layers = Vec::with_capacity(cfg.ratios.len());
        for (i, &ratio) in cfg.ratios.iter().enumerate() {
            let norm = if cfg.disable_norm_outer_blocks + i + 1 >= n_blocks {
                None
            } else {
                Some(cfg.norm)
            };
            let upsample = StreamableConvTranspose1d::load(
                &vb.pp(layer_idx + 1),
                mult * nf,
                mult * nf / 2,
                ratio as i64 * 2,
                ratio as i64,
                1,
                true,
                true,
                norm,
            )?;
            layer_idx += 2;
            let mut residuals = Vec::with_capacity(cfg.n_residual_layers);
            for j in 0..cfg.n_residual_layers {
                let dilation = (cfg.dilation_base as i64).pow(j as u32);
                residuals.push(SeaNetResnetBlock::load(
                    &vb.pp(layer_idx),
                    mult * nf / 2,
                    &[(cfg.residual_kernel_size as i64, dilation), (1, 1)],
                    cfg.activation,
                    norm,
                    cfg.causal,
                    cfg.pad_mode,
                    cfg.compress as i64,
                    cfg.true_skip,
                )?);
                layer_idx += 1;
            }
            layers.push(DecoderLayer { upsample, residuals });
            mult /= 2;
        }

        let final_norm = if cfg.disable_norm_outer_blocks >= 1 { None } else { Some(cfg.norm) };
        let final_conv = StreamableConv1d::load(
            &vb.pp(layer_idx + 1),
            nf,
            cfg.channels as i64,
            cfg.last_kernel_size as i64,
            1,
            1,
            1,
            true,
            cfg.causal,
            final_norm,
            cfg.pad_mode,
        )?;

        Ok(Self {
            init_conv,
            activation: cfg.activation,
            layers,
            final_conv,
            final_activation: cfg.final_activation,
        })
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let mut xs = self.init_conv.forward(xs)?;
        for layer in &self.layers {
            xs = self.activation.apply(&xs)?;
            xs = layer.upsample.forward(&xs)?;
            for residual in &layer.residuals {
                xs = residual.forward(&xs)?;
            }
        }
        xs = self.activation.apply(&xs)?;
        xs = self.final_conv.forward(&xs)?;
        if let Some(act) = &self.final_activation {
            xs = act.apply(&xs)?;
        }
        Ok(xs)
    }

    /// Streaming step: `xs` is `[1, dimension, l]`, returns `[1, channels, l * 960]`.
    pub fn step(&self, xs: &XlaOp, ctx: &mut StepCtx) -> Result<XlaOp> {
        let mut xs = self.init_conv.step(xs, ctx)?;
        for layer in &self.layers {
            xs = self.activation.apply(&xs)?;
            xs = layer.upsample.step(&xs, ctx)?;
            for residual in &layer.residuals {
                xs = residual.step(&xs, ctx)?;
            }
        }
        xs = self.activation.apply(&xs)?;
        xs = self.final_conv.step(&xs, ctx)?;
        if let Some(act) = &self.final_activation {
            xs = act.apply(&xs)?;
        }
        Ok(xs)
    }
}

fn err(msg: &str) -> crate::Error {
    crate::Error::Xla(xla::Error::XlaError { msg: msg.to_string(), backtrace: String::new() })
}
