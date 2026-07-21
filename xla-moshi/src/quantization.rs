//! Residual vector quantization, ported from the `xn-moshi` reference.
use crate::{Result, Vb};
use xla::{ElementType, XlaOp};

/// Contract the last dim of `xs` with the last dim of `w` (i.e. `xs @ w^T`),
/// keeping any leading/batch dims of `xs`.
fn matmul_t(xs: &XlaOp, w: &XlaOp) -> Result<XlaOp> {
    let rank = xs.rank()? as i64;
    Ok(xs.dot_general(w, &[rank - 1], &[1], &[], &[])?)
}

pub struct EuclideanCodebook {
    embedding: XlaOp,
    c2: XlaOp,
    dim: i64,
}

impl EuclideanCodebook {
    pub fn load(vb: &Vb, dim: i64, codebook_size: i64) -> Result<Self> {
        let cluster_usage = vb.var("cluster_usage", &[codebook_size])?;
        let embedding_sum = vb.var("embedding_sum", &[codebook_size, dim])?;
        let b = cluster_usage.builder();
        // embedding = embedding_sum / max(cluster_usage, eps)
        let eps = b.c0(1e-5f32)?;
        let cluster_usage = cluster_usage.max(&eps)?.reshape(&[codebook_size, 1])?;
        let cluster_usage = cluster_usage.broadcast_in_dim(&[codebook_size, dim], &[0, 1])?;
        let embedding = embedding_sum.div_(&cluster_usage)?;
        // c2 = 0.5 * sum(embedding^2, dim=-1)
        let c2 = embedding.mul_(&embedding)?.reduce_sum(&[1], false)?;
        let c2 = c2.mul_(&b.c0(0.5f32)?)?;
        Ok(Self { embedding, c2, dim })
    }

    /// `xs`: `[.., dim]` -> codes `[..]` (S64).
    pub fn encode(&self, xs: &XlaOp) -> Result<XlaOp> {
        let dims = xs.dims()?;
        let mut target: Vec<i64> = dims.iter().map(|d| *d as i64).collect();
        target.pop();
        let n: i64 = target.iter().product::<i64>().max(1);
        let flat = xs.reshape(&[n, self.dim])?;
        // dist = c2 - x . e (the ||x||^2 term is constant and does not shift the argmin).
        let dot = flat.dot_general(&self.embedding, &[1], &[1], &[], &[])?;
        let bins = self.c2.dims()?[0] as i64;
        let c2 = self.c2.broadcast_in_dim(&[n, bins], &[1])?;
        let dists = c2.sub_(&dot)?;
        let codes = dists.argmin(ElementType::S64, 1)?;
        if target.is_empty() {
            Ok(codes)
        } else {
            Ok(codes.reshape(&target)?)
        }
    }

    /// `indices`: `[..]` -> `[.., dim]`.
    pub fn decode(&self, indices: &XlaOp) -> Result<XlaOp> {
        let dims = indices.dims()?;
        let mut final_dims: Vec<i64> = dims.iter().map(|d| *d as i64).collect();
        let n: i64 = final_dims.iter().product::<i64>().max(1);
        final_dims.push(self.dim);
        let flat = indices.reshape(&[n])?;
        let values = self.embedding.take(&flat, 0)?;
        Ok(values.reshape(&final_dims)?)
    }
}

pub struct VectorQuantization {
    project_in: Option<XlaOp>,
    project_out: Option<XlaOp>,
    codebook: EuclideanCodebook,
}

impl VectorQuantization {
    pub fn load(
        vb: &Vb,
        dim: i64,
        codebook_size: i64,
        codebook_dim: Option<i64>,
    ) -> Result<Self> {
        let codebook_dim = codebook_dim.unwrap_or(dim);
        let (project_in, project_out) = if codebook_dim == dim {
            (None, None)
        } else {
            let p_in = vb.pp("project_in").var("weight", &[codebook_dim, dim])?;
            let p_out = vb.pp("project_out").var("weight", &[dim, codebook_dim])?;
            (Some(p_in), Some(p_out))
        };
        let codebook = EuclideanCodebook::load(&vb.pp("_codebook"), codebook_dim, codebook_size)?;
        Ok(Self { project_in, project_out, codebook })
    }

    /// `xs`: `[b, c, t]` -> codes `[b, t]`.
    pub fn encode(&self, xs: &XlaOp) -> Result<XlaOp> {
        let xs = xs.swap_dims(1, 2)?;
        let xs = match &self.project_in {
            Some(p) => matmul_t(&xs, p)?,
            None => xs,
        };
        self.codebook.encode(&xs)
    }

    /// codes `[b, t]` -> `[b, c, t]`.
    pub fn decode(&self, codes: &XlaOp) -> Result<XlaOp> {
        let q = self.codebook.decode(codes)?;
        let q = match &self.project_out {
            Some(p) => matmul_t(&q, p)?,
            None => q,
        };
        Ok(q.swap_dims(1, 2)?)
    }
}

pub struct ResidualVectorQuantization {
    layers: Vec<VectorQuantization>,
}

impl ResidualVectorQuantization {
    pub fn load(
        vb: &Vb,
        n_q: i64,
        dim: i64,
        codebook_size: i64,
        codebook_dim: Option<i64>,
    ) -> Result<Self> {
        let vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(n_q as usize);
        for i in 0..n_q {
            layers.push(VectorQuantization::load(&vb.pp(i), dim, codebook_size, codebook_dim)?);
        }
        Ok(Self { layers })
    }

    /// `xs`: `[b, dim, t]` -> codes `[b, n_q, t]`.
    pub fn encode(&self, xs: &XlaOp) -> Result<XlaOp> {
        let mut residual = xs.clone();
        let mut codes: Vec<XlaOp> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let indices = layer.encode(&residual)?; // [b, t]
            let quantized = layer.decode(&indices)?; // [b, dim, t]
            residual = residual.sub_(&quantized)?;
            let d = indices.dims()?;
            codes.push(indices.reshape(&[d[0] as i64, 1, d[1] as i64])?);
        }
        let first = codes.remove(0);
        if codes.is_empty() {
            Ok(first)
        } else {
            Ok(first.concat_in_dim(&codes, 1)?)
        }
    }

    /// codes `[b, n_q, t]` -> `[b, dim, t]`.
    pub fn decode(&self, codes: &XlaOp) -> Result<XlaOp> {
        let mut quantized: Option<XlaOp> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let d = codes.dims()?;
            let layer_codes =
                codes.slice_in_dim1(i as i64, i as i64 + 1, 1)?.reshape(&[d[0] as i64, d[2] as i64])?;
            let q = layer.decode(&layer_codes)?;
            quantized = Some(match quantized {
                None => q,
                Some(acc) => acc.add_(&q)?,
            });
        }
        quantized.ok_or_else(|| {
            crate::Error::Xla(xla::Error::XlaError {
                msg: "empty layers in ResidualVectorQuantization".to_string(),
                backtrace: String::new(),
            })
        })
    }
}

pub struct ResidualVectorQuantizer {
    vq: ResidualVectorQuantization,
    input_proj: Option<XlaOp>,
    output_proj: Option<XlaOp>,
}

impl ResidualVectorQuantizer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Vb,
        dim: i64,
        input_dim: Option<i64>,
        output_dim: Option<i64>,
        n_q: i64,
        bins: i64,
        force_projection: bool,
    ) -> Result<Self> {
        let input_dim = input_dim.unwrap_or(dim);
        let output_dim = output_dim.unwrap_or(dim);
        let input_proj = if input_dim != dim || force_projection {
            Some(vb.pp("input_proj").var("weight", &[dim, input_dim, 1])?)
        } else {
            None
        };
        let output_proj = if output_dim != dim || force_projection {
            Some(vb.pp("output_proj").var("weight", &[output_dim, dim, 1])?)
        } else {
            None
        };
        let vq = ResidualVectorQuantization::load(&vb.pp("vq"), n_q, dim, bins, None)?;
        Ok(Self { vq, input_proj, output_proj })
    }

    pub fn encode(&self, xs: &XlaOp) -> Result<XlaOp> {
        let xs = match &self.input_proj {
            Some(p) => xs.conv1d(p, 1, 0, 1, 1)?,
            None => xs.clone(),
        };
        self.vq.encode(&xs)
    }

    pub fn decode(&self, codes: &XlaOp) -> Result<XlaOp> {
        let quantized = self.vq.decode(codes)?;
        match &self.output_proj {
            Some(p) => Ok(quantized.conv1d(p, 1, 0, 1, 1)?),
            None => Ok(quantized),
        }
    }
}

pub struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q: i64,
}

impl SplitResidualVectorQuantizer {
    pub fn load(
        vb: &Vb,
        dim: i64,
        input_dim: Option<i64>,
        output_dim: Option<i64>,
        n_q: i64,
        bins: i64,
    ) -> Result<Self> {
        let rvq_first =
            ResidualVectorQuantizer::load(&vb.pp("rvq_first"), dim, input_dim, output_dim, 1, bins, true)?;
        let rvq_rest = ResidualVectorQuantizer::load(
            &vb.pp("rvq_rest"),
            dim,
            input_dim,
            output_dim,
            n_q - 1,
            bins,
            true,
        )?;
        Ok(Self { rvq_first, rvq_rest, n_q })
    }

    /// `xs`: `[b, dim, t]` -> codes `[b, n_q, t]`.
    pub fn encode(&self, xs: &XlaOp) -> Result<XlaOp> {
        let codes = self.rvq_first.encode(xs)?;
        if self.n_q > 1 {
            let rest = self.rvq_rest.encode(xs)?;
            Ok(codes.concat_in_dim(&[rest], 1)?)
        } else {
            Ok(codes)
        }
    }

    /// codes `[b, n_q, t]` -> `[b, dim, t]`.
    pub fn decode(&self, codes: &XlaOp) -> Result<XlaOp> {
        let first_codes = codes.slice_in_dim1(0, 1, 1)?;
        let quantized = self.rvq_first.decode(&first_codes)?;
        if self.n_q > 1 {
            let rest_codes = codes.slice_in_dim1(1, self.n_q, 1)?;
            Ok(quantized.add_(&self.rvq_rest.decode(&rest_codes)?)?)
        } else {
            Ok(quantized)
        }
    }
}
