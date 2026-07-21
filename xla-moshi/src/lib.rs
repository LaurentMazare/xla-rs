//! Moshi / Mimi models built on top of the [`xla`] crate.
//!
//! This currently implements the non-streaming (whole-file) forward path of the
//! Mimi neural audio codec: SEANet encoder/decoder, the Mimi transformer, and
//! the split residual vector quantizer. The implementation mirrors the eager
//! `xn-moshi` reference but builds a single XLA computation for `encode` and one
//! for `decode`.
pub mod conv;
pub mod mimi;
pub mod quantization;
pub mod seanet;
pub mod transformer;

pub use xla_nn::{Error, Result};

use xla::XlaOp;

/// A thin helper that mirrors the `Path` used in the `xn` reference: it keeps a
/// dotted prefix and forwards weight declarations to an [`xla_nn::VarBuilder`],
/// so the safetensors names line up exactly with the reference implementation.
#[derive(Clone)]
pub struct Vb<'a> {
    inner: &'a xla_nn::VarBuilder,
    prefix: String,
}

impl<'a> Vb<'a> {
    pub fn new(inner: &'a xla_nn::VarBuilder) -> Self {
        Self { inner, prefix: String::new() }
    }

    /// Push a component onto the prefix (like `cd`-ing into a directory).
    pub fn pp(&self, s: impl std::fmt::Display) -> Vb<'a> {
        let prefix =
            if self.prefix.is_empty() { s.to_string() } else { format!("{}.{s}", self.prefix) };
        Vb { inner: self.inner, prefix }
    }

    fn key(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }

    /// Declare a weight parameter and return the corresponding graph node.
    pub fn var(&self, name: &str, dims: &[i64]) -> Result<XlaOp> {
        self.inner.var(&self.key(name), dims)
    }

    pub fn var_builder(&self) -> &'a xla_nn::VarBuilder {
        self.inner
    }
}

/// Broadcast-add a per-channel bias of shape `[c]` onto a `[b, c, t]` tensor.
pub(crate) fn add_bias(xs: &XlaOp, bias: &XlaOp) -> Result<XlaOp> {
    let c = bias.dims()?[0] as i64;
    let bias = bias.reshape(&[1, c, 1])?;
    Ok(xs.add_(&bias)?)
}
