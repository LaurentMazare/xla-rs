//! Neural-network helpers built on top of the [`xla`] crate.
//!
//! The main entry point is [`VarBuilder`], which declares model weights as XLA
//! parameters and loads their values from safetensors shards, converting to a
//! target dtype on the host along the way.
pub mod error;
pub mod linear;
pub mod norm;
pub mod var_store;

pub use error::{Error, Result};
pub use linear::Linear;
pub use norm::{LayerNorm, RmsNorm};
pub use var_store::{Path, PleTable, VarBuilder};
