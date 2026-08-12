use crate::{Path, Result};
use xla::XlaOp;

pub struct Linear {
    weight: XlaOp,
    bias: Option<XlaOp>,
}

impl Linear {
    pub fn load(vb: &Path, in_d: i64, out_d: i64, bias: bool) -> Result<Self> {
        let weight = vb.var("weight", &[out_d, in_d])?;
        let bias = if bias { Some(vb.var("bias", &[out_d])?) } else { None };
        Ok(Self { weight, bias })
    }

    pub fn from_weight(weight: XlaOp, bias: Option<XlaOp>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
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
