// A small helper to declare the model weights as XLA parameters and load the
// matching buffers from a safetensors file.
//
// Weights are converted to the target dtype on the host, the buffers are
// created one tensor at a time so that peak host memory stays around a single
// tensor.
use anyhow::{bail, Result};
use xla::{ElementType, PjRtBuffer, PjRtClient, XlaBuilder, XlaOp};

type Vars = std::rc::Rc<std::cell::RefCell<Vec<(String, Vec<i64>)>>>;

pub struct VarBuilder {
    builder: XlaBuilder,
    dtype: ElementType,
    vars: Vars,
    // The first parameter index available for weights, the indices before it
    // are reserved for the non-weight arguments (token ids, position, ...).
    first_weight_index: usize,
}

impl VarBuilder {
    pub fn new(builder: &XlaBuilder, dtype: ElementType, first_weight_index: usize) -> Self {
        Self { builder: builder.clone(), dtype, vars: Default::default(), first_weight_index }
    }

    pub fn dtype(&self) -> ElementType {
        self.dtype
    }

    /// Declare a weight parameter with the given safetensors name.
    pub fn var(&self, name: &str, dims: &[i64]) -> Result<XlaOp> {
        let mut vars = self.vars.borrow_mut();
        let index = vars.len() + self.first_weight_index;
        let op = self.builder.parameter(index as i64, self.dtype, dims, name)?;
        vars.push((name.to_string(), dims.to_vec()));
        Ok(op)
    }

    /// The parameter index right after the last declared weight.
    pub fn next_index(&self) -> usize {
        self.vars.borrow().len() + self.first_weight_index
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

/// A weight kept in host memory as a memory-mapped safetensors tensor, with
/// rows gathered on the cpu and shipped to the device as a computation input.
/// Used for the multi-GB per-layer embedding table which is only ever read a
/// row at a time, so that it does not take up device memory.
pub struct PleTable {
    mmap: memmap2::Mmap,
    data_start: usize,
    rows: usize,
    row_elems: usize,
    dtype: safetensors::Dtype,
    target: ElementType,
}

impl PleTable {
    pub fn load<P: AsRef<std::path::Path>>(
        paths: &[P],
        name: &str,
        dims: &[i64],
        target: ElementType,
    ) -> Result<Self> {
        if dims.len() != 2 {
            bail!("expected a rank 2 shape for {name}, got {dims:?}")
        }
        for path in paths.iter() {
            let file = std::fs::File::open(path.as_ref())?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let found = {
                let st = safetensors::SafeTensors::deserialize(&mmap)?;
                match st.tensor(name) {
                    Err(_) => None,
                    Ok(view) => {
                        let view_dims: Vec<i64> = view.shape().iter().map(|d| *d as i64).collect();
                        if view_dims != *dims {
                            bail!("shape mismatch for {name}: expected {dims:?}, got {view_dims:?}")
                        }
                        let data_start = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
                        Some((data_start, view.dtype()))
                    }
                }
            };
            if let Some((data_start, dtype)) = found {
                dtype_size(name, dtype)?;
                return Ok(Self {
                    mmap,
                    data_start,
                    rows: dims[0] as usize,
                    row_elems: dims[1] as usize,
                    dtype,
                    target,
                });
            }
        }
        bail!("cannot find tensor {name} in the shards")
    }

    /// Gather the rows for the given token ids into a device buffer of shape
    /// [ids.len(), row_elems], converted to the target dtype.
    pub fn gather_buffer(&self, client: &PjRtClient, ids: &[i32]) -> Result<PjRtBuffer> {
        let row_bytes = self.row_elems * dtype_size("ple table", self.dtype)?;
        let mut data = Vec::with_capacity(ids.len() * row_bytes);
        for &id in ids.iter() {
            let id = id as usize;
            if id >= self.rows {
                bail!("token id {id} out of range for the ple table ({} rows)", self.rows)
            }
            let start = self.data_start + id * row_bytes;
            data.extend_from_slice(&self.mmap[start..start + row_bytes]);
        }
        let dims = [ids.len(), self.row_elems];
        let src_matches = matches!(
            (self.dtype, self.target),
            (safetensors::Dtype::F32, ElementType::F32)
                | (safetensors::Dtype::BF16, ElementType::Bf16)
                | (safetensors::Dtype::F16, ElementType::F16)
        );
        let buffer = if src_matches {
            client.buffer_from_host_raw_bytes(self.target, &data, &dims, None)?
        } else {
            let data = to_f32_vec("ple table", self.dtype, &data)?;
            match self.target {
                ElementType::F32 => client.buffer_from_host_buffer(&data, &dims, None)?,
                ElementType::Bf16 => {
                    let data: Vec<u8> =
                        data.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
                    client.buffer_from_host_raw_bytes(self.target, &data, &dims, None)?
                }
                ElementType::F16 => {
                    let data: Vec<u8> =
                        data.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
                    client.buffer_from_host_raw_bytes(self.target, &data, &dims, None)?
                }
                dtype => bail!("unsupported target dtype {dtype:?}"),
            }
        };
        Ok(buffer)
    }
}

fn dtype_size(name: &str, dtype: safetensors::Dtype) -> Result<usize> {
    match dtype {
        safetensors::Dtype::F32 => Ok(4),
        safetensors::Dtype::BF16 | safetensors::Dtype::F16 => Ok(2),
        dtype => bail!("unsupported dtype {dtype:?} for {name}"),
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
