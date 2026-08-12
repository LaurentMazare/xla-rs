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
        // Two-pass variance: the `E[x^2] - E[x]^2` form cancels
        // catastrophically when the mean dwarfs the standard deviation. The
        // variance is computed around the mean even when the mean is not
        // removed from the normalized value, as in the `xn-core` reference.
        let centered = (&xs - &mean)?;
        let var = (&centered * &centered)?.reduce_mean(&[-1], true)?;
        let mul = (var + b.c0(self.eps)?)?.rsqrt()?;
        let xs = if self.remove_mean { centered } else { xs };
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

#[cfg(test)]
mod tests {
    use super::*;
    use xla::XlaBuilder;

    /// Run a no-input computation on the CPU client and read every tuple
    /// element back as f32s.
    fn run(build: impl FnOnce(&XlaBuilder) -> Result<Vec<XlaOp>>) -> Result<Vec<Vec<f32>>> {
        let client = xla::PjRtClient::cpu()?;
        let builder = XlaBuilder::new("test");
        let outs = build(&builder)?;
        let root = builder.tuple(&outs.iter().collect::<Vec<_>>())?;
        let exe = client.compile(&root.build()?)?;
        let bufs = exe.execute::<xla::Literal>(&[])?;
        let bufs = bufs.into_iter().next().expect("no execution output");
        bufs.iter().map(|b| Ok(b.to_literal_sync()?.to_vec::<f32>()?)).collect()
    }

    /// `LayerNorm` on a bf16 activation whose mean dwarfs its standard
    /// deviation — the case where the single-pass `E[x^2] - E[x]^2` variance
    /// cancels catastrophically (with d = 3072 and x around 100, `E[x^2]` is
    /// around 1e4 while the true variance is about 1). This pins the f32
    /// two-pass form.
    #[test]
    fn layer_norm_is_stable_for_bf16_inputs_with_a_large_mean() -> Result<()> {
        const D: usize = 3072;
        let xs: Vec<f32> = (0..D).map(|i| 100.0 + ((i % 7) as f32 - 3.0) * 0.5).collect();
        let outs = run(|b| {
            let d = D as i64;
            // Round the input through bf16 so the reference below sees exactly
            // the values the graph does.
            let xs = b.constant_r1(&xs)?.reshape(&[1, 1, d])?.convert(PrimitiveType::Bf16)?;
            let weight = b.constant_r1c(1f32, D)?.convert(PrimitiveType::Bf16)?;
            let bias = b.constant_r1c(0f32, D)?.convert(PrimitiveType::Bf16)?;
            let norm = LayerNorm::new(weight, bias, 1e-5)?;
            let out = norm.forward(&xs)?;
            Ok(vec![out.convert(PrimitiveType::F32)?, xs.convert(PrimitiveType::F32)?])
        })?;
        let (got, xs_bf16) = (&outs[0], &outs[1]);

        // Reference: two-pass layer norm in f64 over the bf16-rounded input.
        let n = xs_bf16.len() as f64;
        let mean = xs_bf16.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var = xs_bf16.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + 1e-5).sqrt();

        assert_eq!(got.len(), D);
        let mut max_err = 0f64;
        for (g, x) in got.iter().zip(xs_bf16.iter()) {
            assert!(g.is_finite(), "non-finite output {g}");
            max_err = max_err.max((*g as f64 - (*x as f64 - mean) * inv).abs());
        }
        // The output is unit-variance, so this is an absolute tolerance on a
        // quantity of order 1. The unstable form misses by whole multiples.
        assert!(max_err < 0.02, "layer_norm max error {max_err} is too large");
        Ok(())
    }
}
