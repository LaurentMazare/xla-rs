//! Layer-norm and rms-norm helpers, mirroring `xn-core`'s `nn::norm`.
//!
//! Both norms are computed in f32 with the learned parameters also applied in
//! f32, the result is cast back to the input dtype at the end.
use crate::{Path, Result};
use xla::{PrimitiveType, XlaOp};

/// Convert a `[d]` parameter to f32 and broadcast it against the last
/// dimension of `dims`.
fn broadcast_last(w: &XlaOp, dims: &[i64]) -> Result<XlaOp> {
    let rank = dims.len() as i64;
    Ok(w.convert(PrimitiveType::F32)?.broadcast_in_dim(dims, &[rank - 1])?)
}

pub struct RmsNorm {
    weight: XlaOp,
    eps: f32,
}

impl RmsNorm {
    /// `weight` is the learned scale, of shape `[d]`.
    pub fn new(weight: XlaOp, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Load the scale from the `weight` tensor.
    pub fn load(vb: &Path, dim: i64, eps: f32) -> Result<Self> {
        Ok(Self::new(vb.var("weight", &[dim])?, eps))
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let b = xs.builder();
        let dt = xs.ty()?;
        let xs = xs.convert(PrimitiveType::F32)?;
        let mean2 = ((&xs * &xs)?.reduce_mean(&[-1], true)? + b.c0(self.eps)?)?;
        let xs_norm = (&xs * mean2.rsqrt()?)?;
        let shape = xs.array_shape()?;
        let weight = broadcast_last(&self.weight, shape.dims())?;
        Ok((xs_norm * weight)?.convert(dt)?)
    }
}

pub struct LayerNorm {
    weight: XlaOp,
    bias: XlaOp,
    hidden_size: i64,
    remove_mean: bool,
    unbiased: bool,
    eps: f32,
}

impl LayerNorm {
    /// `weight` and `bias` are the learned scale and offset, of shape `[d]`.
    pub fn new(weight: XlaOp, bias: XlaOp, eps: f32) -> Result<Self> {
        let hidden_size = weight.dims()?[0] as i64;
        Ok(Self { weight, bias, eps, hidden_size, unbiased: false, remove_mean: true })
    }

    /// Whether the mean is removed from the normalized value (defaults to
    /// true). The variance is computed around the mean either way.
    pub fn remove_mean(mut self, remove_mean: bool) -> Self {
        self.remove_mean = remove_mean;
        self
    }

    /// Use the unbiased variance estimate, scaling the normalized value by
    /// `sqrt((d - 1) / d)` (defaults to false).
    pub fn unbiased(mut self, unbiased: bool) -> Self {
        self.unbiased = unbiased;
        self
    }

    /// Load the scale and offset from the `weight` and `bias` tensors.
    pub fn load(vb: &Path, dim: i64, eps: f32) -> Result<Self> {
        Self::new(vb.var("weight", &[dim])?, vb.var("bias", &[dim])?, eps)
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let b = xs.builder();
        let dt = xs.ty()?;
        let xs = xs.convert(PrimitiveType::F32)?;
        let mean = xs.reduce_mean(&[-1], true)?;
        let mean2 = (&xs * &xs)?.reduce_mean(&[-1], true)?;
        let var = (mean2 - (&mean * &mean)?)?;
        let mul = (var + b.c0(self.eps)?)?.rsqrt()?;
        let xs = if self.remove_mean { (&xs - &mean)? } else { xs };
        let shape = xs.array_shape()?;
        let weight = broadcast_last(&self.weight, shape.dims())?;
        let bias = broadcast_last(&self.bias, shape.dims())?;
        let ln = (bias + (xs * mul)? * weight)?.convert(dt)?;
        let ln = if self.unbiased {
            let hidden_size = self.hidden_size as f32;
            let s = ((hidden_size - 1f32) / hidden_size).sqrt();
            (ln * b.c0(s)?.convert(dt)?)?
        } else {
            ln
        };
        Ok(ln)
    }
}
