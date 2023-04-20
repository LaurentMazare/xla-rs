/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Incorrect number of elements.
    #[error("wrong element count {element_count} for dims {dims:?}")]
    WrongElementCount { dims: Vec<usize>, element_count: usize },

    /// Error from the xla C++ library.
    #[error("xla error {msg}\n{backtrace}")]
    XlaError { msg: String, backtrace: String },

    #[error("unexpected element type {0}")]
    UnexpectedElementType(i32),

    #[error("unexpected number of dimensions, expected: {expected}, got: {got} ({dims:?})")]
    UnexpectedNumberOfDims { expected: usize, got: usize, dims: Vec<i64> },

    #[error("element type mismatch, on-device: {on_device:?}, on-host: {on_host:?}")]
    ElementTypeMismatch { on_device: crate::PrimitiveType, on_host: crate::PrimitiveType },

    #[error("unsupported element type for {op}: {ty:?}")]
    UnsupportedElementType { ty: crate::PrimitiveType, op: &'static str },

    #[error(
        "target buffer is too large, offset {offset}, shape {shape:?}, buffer_len: {buffer_len}"
    )]
    TargetBufferIsTooLarge { offset: usize, shape: crate::Shape, buffer_len: usize },

    #[error("binary buffer is too large, element count {element_count}, buffer_len: {buffer_len}")]
    BinaryBufferIsTooLarge { element_count: usize, buffer_len: usize },

    #[error("empty literal")]
    EmptyLiteral,

    #[error("index out of bounds {index}, rank {rank}")]
    IndexOutOfBounds { index: i64, rank: usize },

    #[error("npy/npz error {0}")]
    Npy(String),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Zip file format error.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("cannot create literal with shape {element_type:?} {dims:?} from bytes data with len {data_len_in_bytes}")]
    CannotCreateLiteralWithData {
        data_len_in_bytes: usize,
        element_type: crate::PrimitiveType,
        dims: Vec<usize>,
    },

    #[error("invalid dimensions in matmul, lhs: {lhs_dims:?}, rhs: {rhs_dims:?}, {msg}")]
    MatMulIncorrectDims { lhs_dims: Vec<i64>, rhs_dims: Vec<i64>, msg: &'static str },
}

pub type Result<T> = std::result::Result<T, Error>;
