/// Error type for the xla-nn helpers.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// A tensor declared as a weight could not be found in any of the shards.
    #[error("cannot find tensor {name} in the shards")]
    TensorNotFound { name: String },

    /// The on-disk shape does not match the shape the weight was declared with.
    #[error("shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    ShapeMismatch { name: String, expected: Vec<i64>, got: Vec<i64> },

    /// The on-disk dtype cannot be handled by the loader.
    #[error("unsupported source dtype {dtype:?} for {name}")]
    UnsupportedSourceDType { name: String, dtype: safetensors::Dtype },

    /// The requested target element type cannot be produced by the loader.
    #[error("unsupported target dtype {dtype:?}")]
    UnsupportedTargetDType { dtype: xla::ElementType },

    /// A rank-2 shape was expected (e.g. for an embedding table).
    #[error("expected a rank 2 shape for {name}, got {dims:?}")]
    ExpectedRank2 { name: String, dims: Vec<i64> },

    /// A gathered row index falls outside the table.
    #[error("token id {id} out of range for the table ({rows} rows)")]
    IndexOutOfRange { id: usize, rows: usize },

    /// Some tensors present in the shards were never used.
    #[error("{} unused tensors {names:?}", names.len())]
    UnusedTensors { names: Vec<String> },

    /// Error from the underlying xla crate.
    #[error(transparent)]
    Xla(#[from] xla::Error),

    /// Error while parsing a safetensors file.
    #[error(transparent)]
    SafeTensor(#[from] safetensors::SafeTensorError),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
