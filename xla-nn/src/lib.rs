//! Neural-network helpers built on top of the [`xla`] crate.
//!
//! The main entry point is [`VarBuilder`], which declares model weights as XLA
//! parameters and loads their values from safetensors shards, converting to a
//! target dtype on the host along the way.
pub mod error;
pub mod var_store;

pub use error::{Error, Result};
pub use var_store::{PleTable, VarBuilder};
