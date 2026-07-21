//! The Mimi neural audio codec, ported from `xn-moshi`. Provides the
//! non-streaming (whole-file) `encode` and `decode` graph builders.
use crate::conv::{ConvDownsample1d, ConvTrUpsample1d, Norm, PadMode};
use crate::quantization::SplitResidualVectorQuantizer;
use crate::seanet::{self, SeaNetDecoder, SeaNetEncoder};
use crate::transformer::{self, PositionalEmbedding, ProjectedTransformer};
use crate::{Result, Vb};
use xla::XlaOp;

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResampleMethod {
    Conv,
    Interpolate,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub channels: usize,
    pub sample_rate: f64,
    pub frame_rate: f64,
    pub renormalize: bool,
    pub resample_method: ResampleMethod,
    pub seanet: seanet::Config,
    pub transformer: transformer::Config,
    pub quantizer_n_q: usize,
    pub quantizer_bins: usize,
    pub quantizer_dim: usize,
}

impl Config {
    pub fn v0_1(num_codebooks: Option<usize>) -> Self {
        let seanet = seanet::Config {
            dimension: 512,
            channels: 1,
            causal: true,
            n_filters: 64,
            n_residual_layers: 1,
            activation: seanet::Activation::Elu(1.),
            compress: 2,
            dilation_base: 2,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: 7,
            residual_kernel_size: 3,
            last_kernel_size: 3,
            lstm: None,
            norm: Norm::WeightNorm,
            pad_mode: PadMode::Constant,
            ratios: vec![8, 6, 5, 4],
            true_skip: true,
        };
        let transformer = transformer::Config {
            d_model: seanet.dimension,
            num_heads: 8,
            num_layers: 8,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: Some(0.01),
            context: 250,
            use_conv_block: false,
            max_period: 10000.0,
            positional_embedding: PositionalEmbedding::Rope,
            dim_feedforward: 2048,
            kv_repeat: 1,
            conv_layout: true,
        };
        Config {
            channels: 1,
            sample_rate: 24_000.,
            frame_rate: 12.5,
            renormalize: true,
            resample_method: ResampleMethod::Conv,
            seanet,
            transformer,
            quantizer_n_q: num_codebooks.unwrap_or(16),
            quantizer_bins: 2048,
            quantizer_dim: 256,
        }
    }
}

pub struct Mimi {
    encoder: SeaNetEncoder,
    decoder: SeaNetDecoder,
    encoder_transformer: ProjectedTransformer,
    decoder_transformer: ProjectedTransformer,
    downsample: ConvDownsample1d,
    upsample: ConvTrUpsample1d,
    quantizer: SplitResidualVectorQuantizer,
    config: Config,
}

impl Mimi {
    pub fn load(vb: &Vb, cfg: Config) -> Result<Self> {
        let dim = cfg.seanet.dimension as i64;
        let encoder = SeaNetEncoder::load(&vb.pp("encoder"), &cfg.seanet)?;
        let decoder = SeaNetDecoder::load(&vb.pp("decoder"), &cfg.seanet)?;
        let encoder_transformer =
            ProjectedTransformer::load(&vb.pp("encoder_transformer"), dim, &cfg.transformer)?;
        let decoder_transformer =
            ProjectedTransformer::load(&vb.pp("decoder_transformer"), dim, &cfg.transformer)?;
        let quantizer = SplitResidualVectorQuantizer::load(
            &vb.pp("quantizer"),
            cfg.quantizer_dim as i64,
            Some(dim),
            Some(dim),
            cfg.quantizer_n_q as i64,
            cfg.quantizer_bins as i64,
        )?;
        let encoder_frame_rate =
            cfg.sample_rate / cfg.seanet.ratios.iter().product::<usize>() as f64;
        let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as i64;
        let downsample = ConvDownsample1d::load(&vb.pp("downsample"), downsample_stride, dim, true)?;
        let upsample = ConvTrUpsample1d::load(&vb.pp("upsample"), downsample_stride, dim, true)?;
        Ok(Self {
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            downsample,
            upsample,
            quantizer,
            config: cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Encode audio `[batch, 1, len]` to integer codes `[batch, n_q, frames]`.
    pub fn encode(&self, xs: &XlaOp) -> Result<XlaOp> {
        let xs = self.encoder.forward(xs)?;
        let xs = self.encoder_transformer.forward(&xs)?;
        let xs = self.downsample.forward(&xs)?;
        self.quantizer.encode(&xs)
    }

    /// Decode integer codes `[batch, n_q, frames]` to audio `[batch, 1, len]`.
    pub fn decode(&self, codes: &XlaOp) -> Result<XlaOp> {
        let emb = self.quantizer.decode(codes)?;
        let emb = self.upsample.forward(&emb)?;
        let outs = self.decoder_transformer.forward(&emb)?;
        self.decoder.forward(&outs)
    }

    /// Encoder + transformer + downsample, before quantization.
    pub fn encode_pre_quantize(&self, xs: &XlaOp) -> Result<XlaOp> {
        let xs = self.encoder.forward(xs)?;
        let xs = self.encoder_transformer.forward(&xs)?;
        self.downsample.forward(&xs)
    }

    /// Debug: per-stage decode tensors (emb, upsampled, post-transformer, final).
    pub fn decode_stages(&self, codes: &XlaOp) -> Result<Vec<XlaOp>> {
        let emb = self.quantizer.decode(codes)?;
        let up = self.upsample.forward(&emb)?;
        let dtf = self.decoder_transformer.forward(&up)?;
        let out = self.decoder.forward(&dtf)?;
        Ok(vec![emb, up, dtf, out])
    }
}
