/// Main library error type.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Incorrect number of elements.
    #[error("wrong element count {element_count} for dims {dims:?}")]
    WrongElementCount { dims: Vec<usize>, element_count: usize },

    /// Error from the xla C++ library.
    #[error("xla error {0}")]
    XlaError(String),

    #[error("unexpected element type {0}")]
    UnexpectedElementType(i32),
}

pub type Result<T> = std::result::Result<T, Error>;
