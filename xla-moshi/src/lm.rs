//! The Moshi speech-to-text language model, mirroring `xn-moshi`'s `lm.rs`.
//!
//! The model sums a text-token embedding with one embedding per audio codebook
//! and runs a causal transformer over the resulting stream, one 12.5Hz frame
//! at a time. [`LmModel::step`] builds a single-step computation: the previous
//! text token plus one slice of audio codes in, the greedy next text token
//! out, with the per-layer kv caches threaded through a [`StepCtx`].
use crate::transformer::{self, Norm, Transformer};
use crate::{Result, StepCtx, Vb};
use xla::{ElementType, PrimitiveType, XlaOp};

#[derive(Debug, Clone)]
pub struct Config {
    pub transformer: transformer::Config,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
}

impl Config {
    /// The kyutai/stt-2.6b-en configuration.
    pub fn stt_2_6b() -> Self {
        let transformer = transformer::Config {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 8448, // 2048 * 4.125
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 375,
            max_period: 100_000.0,
            use_conv_block: false,
            gating: Some(crate::seanet::Activation::Silu),
            norm: transformer::NormType::RmsNorm,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
            ring_kv_cache: true,
        };
        Self {
            transformer,
            audio_vocab_size: 2049,
            text_in_vocab_size: 4001,
            text_out_vocab_size: 4000,
            audio_codebooks: 32,
        }
    }
}

pub struct LmModel {
    transformer: Transformer,
    text_emb: XlaOp,        // [text_in_vocab_size, d_model]
    audio_embs: Vec<XlaOp>, // each [audio_vocab_size, d_model]
    text_linear: XlaOp,     // [text_out_vocab_size, d_model]
    out_norm: Norm,
    audio_vocab_size: usize,
    text_in_vocab_size: usize,
}

impl LmModel {
    pub fn load(vb: &Vb, cfg: &Config) -> Result<Self> {
        let d_model = cfg.transformer.d_model as i64;
        let text_emb =
            vb.pp("text_emb").var("weight", &[cfg.text_in_vocab_size as i64, d_model])?;
        let out_norm = Norm::load(&vb.pp("out_norm"), d_model, cfg.transformer.norm)?;
        let text_linear =
            vb.pp("text_linear").var("weight", &[cfg.text_out_vocab_size as i64, d_model])?;
        let transformer = Transformer::load(&vb.pp("transformer"), &cfg.transformer)?;
        let vb_e = vb.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.audio_codebooks);
        for i in 0..cfg.audio_codebooks {
            audio_embs.push(vb_e.pp(i).var("weight", &[cfg.audio_vocab_size as i64, d_model])?);
        }
        Ok(Self {
            transformer,
            text_emb,
            audio_embs,
            text_linear,
            out_norm,
            audio_vocab_size: cfg.audio_vocab_size,
            text_in_vocab_size: cfg.text_in_vocab_size,
        })
    }

    /// The padding token used in place of the audio codes on the first step.
    pub fn audio_pad_token(&self) -> i64 {
        self.audio_vocab_size as i64 - 1
    }

    /// The text token fed on the first step.
    pub fn text_start_token(&self) -> i32 {
        self.text_in_vocab_size as i32 - 1
    }

    /// One decoding step. `text_token`: `[b]` (s32), the previous text tokens.
    /// `audio_codes`: `[b, codebooks, 1]` (s64), one slice of Mimi codes.
    /// Returns the greedy next text tokens `[b]` (s32).
    pub fn step(
        &self,
        text_token: &XlaOp,
        audio_codes: &XlaOp,
        ctx: &mut StepCtx,
    ) -> Result<XlaOp> {
        let b = text_token.dims()?[0] as i64;
        let d_model = self.text_emb.dims()?[1] as i64;
        let mut emb = self.text_emb.take(text_token, 0)?.reshape(&[b, 1, d_model])?;
        for (i, audio_emb) in self.audio_embs.iter().enumerate() {
            let id = audio_codes
                .slice_in_dim1(i as i64, i as i64 + 1, 1)?
                .reshape(&[b])?
                .convert(PrimitiveType::S32)?;
            let e = audio_emb.take(&id, 0)?.reshape(&[b, 1, d_model])?;
            emb = emb.add_(&e)?;
        }
        let ys = self.transformer.step(&emb, ctx)?;
        let ys = self.out_norm.forward(&ys)?;
        let rank = ys.rank()? as i64;
        let logits = ys.dot_general(&self.text_linear, &[rank - 1], &[1], &[], &[])?;
        // Greedy sampling: the reference uses gumbel_max which degenerates to
        // argmax at temperature 0 (computed over f32 logits).
        Ok(logits
            .reshape(&[b, logits.dims()?[2] as i64])?
            .convert(PrimitiveType::F32)?
            .argmax(ElementType::S32, -1)?
            .reshape(&[b])?)
    }
}
