// A small helper to declare the model weights as XLA parameters and load the
// matching buffers from a safetensors file.
//
// Weights are converted to f32 on the host, the buffers are created one tensor
// at a time so that peak host memory stays around a single tensor.
use anyhow::{bail, Result};
use xla::{ElementType, PjRtBuffer, PjRtClient, XlaBuilder, XlaOp};

// Parameters 0 and 1 are reserved for the token ids and the last position.
pub const NUM_NON_WEIGHT_ARGS: usize = 2;

type Vars = std::rc::Rc<std::cell::RefCell<Vec<(String, Vec<i64>)>>>;

pub struct VarBuilder {
    builder: XlaBuilder,
    vars: Vars,
}

impl VarBuilder {
    pub fn new(builder: &XlaBuilder) -> Self {
        Self { builder: builder.clone(), vars: Default::default() }
    }

    /// Declare a f32 weight parameter with the given safetensors name.
    pub fn var(&self, name: &str, dims: &[i64]) -> Result<XlaOp> {
        let mut vars = self.vars.borrow_mut();
        let index = vars.len() + NUM_NON_WEIGHT_ARGS;
        let op = self.builder.parameter(index as i64, ElementType::F32, dims, name)?;
        vars.push((name.to_string(), dims.to_vec()));
        Ok(op)
    }

    /// The number of declared weight parameters.
    pub fn num_vars(&self) -> usize {
        self.vars.borrow().len()
    }

    /// Load the declared weights from a safetensors file, in declaration order.
    pub fn load_buffers<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        client: &PjRtClient,
    ) -> Result<Vec<PjRtBuffer>> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;
        let vars = self.vars.borrow();
        let mut buffers = Vec::with_capacity(vars.len());
        for (name, dims) in vars.iter() {
            let view = st.tensor(name)?;
            let view_dims: Vec<i64> = view.shape().iter().map(|d| *d as i64).collect();
            if view_dims != *dims {
                bail!("shape mismatch for {name}: expected {dims:?}, got {view_dims:?}")
            }
            let data = to_f32_vec(name, view.dtype(), view.data())?;
            let dims_usize: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
            let buffer = client.buffer_from_host_buffer(&data, &dims_usize, None)?;
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
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        dtype => bail!("unsupported dtype {dtype:?} for {name}"),
    };
    Ok(res)
}
