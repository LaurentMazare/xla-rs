// A small helper to declare the model weights as XLA parameters and load the
// matching buffers from a safetensors file.
//
// Weights are converted to the target dtype on the host, the buffers are
// created one tensor at a time so that peak host memory stays around a single
// tensor.
use anyhow::{bail, Result};
use xla::{ElementType, PjRtBuffer, PjRtClient, XlaBuilder, XlaOp};

// Parameters 0 and 1 are reserved for the token ids and the last position.
pub const NUM_NON_WEIGHT_ARGS: usize = 2;

type Vars = std::rc::Rc<std::cell::RefCell<Vec<(String, Vec<i64>)>>>;

pub struct VarBuilder {
    builder: XlaBuilder,
    dtype: ElementType,
    vars: Vars,
}

impl VarBuilder {
    pub fn new(builder: &XlaBuilder, dtype: ElementType) -> Self {
        Self { builder: builder.clone(), dtype, vars: Default::default() }
    }

    pub fn dtype(&self) -> ElementType {
        self.dtype
    }

    /// Declare a weight parameter with the given safetensors name.
    pub fn var(&self, name: &str, dims: &[i64]) -> Result<XlaOp> {
        let mut vars = self.vars.borrow_mut();
        let index = vars.len() + NUM_NON_WEIGHT_ARGS;
        let op = self.builder.parameter(index as i64, self.dtype, dims, name)?;
        vars.push((name.to_string(), dims.to_vec()));
        Ok(op)
    }

    /// The number of declared weight parameters.
    pub fn num_vars(&self) -> usize {
        self.vars.borrow().len()
    }

    /// Load the declared weights from safetensors shards, in declaration order.
    pub fn load_buffers<P: AsRef<std::path::Path>>(
        &self,
        paths: &[P],
        client: &PjRtClient,
    ) -> Result<Vec<PjRtBuffer>> {
        let mut mmaps = Vec::with_capacity(paths.len());
        for path in paths.iter() {
            let file = std::fs::File::open(path.as_ref())?;
            mmaps.push(unsafe { memmap2::Mmap::map(&file)? });
        }
        let sts = mmaps
            .iter()
            .map(|m| safetensors::SafeTensors::deserialize(m))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let vars = self.vars.borrow();
        let mut buffers = Vec::with_capacity(vars.len());
        for (name, dims) in vars.iter() {
            let view = sts
                .iter()
                .find_map(|st| st.tensor(name).ok())
                .ok_or_else(|| anyhow::anyhow!("cannot find tensor {name} in the shards"))?;
            let view_dims: Vec<i64> = view.shape().iter().map(|d| *d as i64).collect();
            if view_dims != *dims {
                bail!("shape mismatch for {name}: expected {dims:?}, got {view_dims:?}")
            }
            let dims_usize: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
            let src_matches = matches!(
                (view.dtype(), self.dtype),
                (safetensors::Dtype::F32, ElementType::F32)
                    | (safetensors::Dtype::BF16, ElementType::Bf16)
                    | (safetensors::Dtype::F16, ElementType::F16)
            );
            let buffer = if src_matches {
                // Same source and target dtype, pass the raw bytes through.
                client.buffer_from_host_raw_bytes(self.dtype, view.data(), &dims_usize, None)?
            } else {
                let data = to_f32_vec(name, view.dtype(), view.data())?;
                match self.dtype {
                    ElementType::F32 => client.buffer_from_host_buffer(&data, &dims_usize, None)?,
                    ElementType::Bf16 => {
                        let data: Vec<u8> = data
                            .iter()
                            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                            .collect();
                        client.buffer_from_host_raw_bytes(self.dtype, &data, &dims_usize, None)?
                    }
                    ElementType::F16 => {
                        let data: Vec<u8> = data
                            .iter()
                            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                            .collect();
                        client.buffer_from_host_raw_bytes(self.dtype, &data, &dims_usize, None)?
                    }
                    dtype => bail!("unsupported target dtype {dtype:?}"),
                }
            };
            buffers.push(buffer)
        }
        Ok(buffers)
    }
}

fn to_f32_vec(name: &str, dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>> {
    let res = match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
        }
        safetensors::Dtype::F16 => {
            data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
        }
        dtype => bail!("unsupported dtype {dtype:?} for {name}"),
    };
    Ok(res)
}
