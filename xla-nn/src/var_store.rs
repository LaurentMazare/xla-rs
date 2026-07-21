// A small helper to declare model weights as XLA parameters and load the
// matching buffers from safetensors files.
//
// Weights are converted to the target dtype on the host, and the buffers are
// created one tensor at a time so that peak host memory stays around a single
// tensor.
use crate::error::{Error, Result};
use xla::{ElementType, PjRtBuffer, PjRtClient, XlaBuilder, XlaOp};

type Vars = std::rc::Rc<std::cell::RefCell<Vec<(String, Vec<i64>)>>>;

/// Declares model weights as XLA parameters and loads their values from
/// safetensors shards, in declaration order.
pub struct VarBuilder {
    builder: XlaBuilder,
    dtype: ElementType,
    vars: Vars,
    // The first parameter index available for weights; the indices before it
    // are reserved for the non-weight arguments (token ids, position, ...).
    first_weight_index: usize,
}

impl VarBuilder {
    /// Create a new builder. `first_weight_index` is the number of parameters
    /// reserved for the non-weight arguments, so the first declared weight gets
    /// that parameter index.
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

    /// The number of declared weight parameters.
    pub fn num_vars(&self) -> usize {
        self.vars.borrow().len()
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
                .ok_or_else(|| Error::TensorNotFound { name: name.clone() })?;
            let view_dims: Vec<i64> = view.shape().iter().map(|d| *d as i64).collect();
            if view_dims != *dims {
                return Err(Error::ShapeMismatch {
                    name: name.clone(),
                    expected: dims.clone(),
                    got: view_dims,
                });
            }
            let dims_usize: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
            let buffer =
                buffer_from_view(client, name, view.dtype(), view.data(), &dims_usize, self.dtype)?;
            buffers.push(buffer)
        }
        Ok(buffers)
    }
}

/// A weight kept in host memory as a memory-mapped safetensors tensor, with
/// rows gathered on the cpu and shipped to the device as a computation input.
/// Useful for a multi-GB embedding table which is only ever read a row at a
/// time, so that it does not take up device memory.
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
            return Err(Error::ExpectedRank2 { name: name.to_string(), dims: dims.to_vec() });
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
                            return Err(Error::ShapeMismatch {
                                name: name.to_string(),
                                expected: dims.to_vec(),
                                got: view_dims,
                            });
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
        Err(Error::TensorNotFound { name: name.to_string() })
    }

    /// Gather the rows for the given token ids into a device buffer of shape
    /// [ids.len(), row_elems], converted to the target dtype.
    pub fn gather_buffer(&self, client: &PjRtClient, ids: &[i32]) -> Result<PjRtBuffer> {
        let row_bytes = self.row_elems * dtype_size("ple table", self.dtype)?;
        let mut data = Vec::with_capacity(ids.len() * row_bytes);
        for &id in ids.iter() {
            let id = id as usize;
            if id >= self.rows {
                return Err(Error::IndexOutOfRange { id, rows: self.rows });
            }
            let start = self.data_start + id * row_bytes;
            data.extend_from_slice(&self.mmap[start..start + row_bytes]);
        }
        let dims = [ids.len(), self.row_elems];
        buffer_from_view(client, "ple table", self.dtype, &data, &dims, self.target)
    }
}

/// Create a device buffer from raw safetensors bytes, converting to `target`
/// if the source dtype differs.
fn buffer_from_view(
    client: &PjRtClient,
    name: &str,
    src: safetensors::Dtype,
    data: &[u8],
    dims: &[usize],
    target: ElementType,
) -> Result<PjRtBuffer> {
    let src_matches = matches!(
        (src, target),
        (safetensors::Dtype::F32, ElementType::F32)
            | (safetensors::Dtype::BF16, ElementType::Bf16)
            | (safetensors::Dtype::F16, ElementType::F16)
    );
    let buffer = if src_matches {
        // Same source and target dtype, pass the raw bytes through.
        client.buffer_from_host_raw_bytes(target, data, dims, None)?
    } else {
        let data = to_f32_vec(name, src, data)?;
        match target {
            ElementType::F32 => client.buffer_from_host_buffer(&data, dims, None)?,
            ElementType::Bf16 => {
                let data: Vec<u8> =
                    data.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
                client.buffer_from_host_raw_bytes(target, &data, dims, None)?
            }
            ElementType::F16 => {
                let data: Vec<u8> =
                    data.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()).collect();
                client.buffer_from_host_raw_bytes(target, &data, dims, None)?
            }
            dtype => return Err(Error::UnsupportedTargetDType { dtype }),
        }
    };
    Ok(buffer)
}

fn dtype_size(name: &str, dtype: safetensors::Dtype) -> Result<usize> {
    match dtype {
        safetensors::Dtype::F32 => Ok(4),
        safetensors::Dtype::BF16 | safetensors::Dtype::F16 => Ok(2),
        dtype => Err(Error::UnsupportedSourceDType { name: name.to_string(), dtype }),
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
        dtype => return Err(Error::UnsupportedSourceDType { name: name.to_string(), dtype }),
    };
    Ok(res)
}
