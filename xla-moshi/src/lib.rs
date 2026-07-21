//! Moshi / Mimi models built on top of the [`xla`] crate.
//!
//! This implements the Mimi neural audio codec (SEANet encoder/decoder, the Mimi
//! transformer, and the split residual vector quantizer), mirroring the eager
//! `xn-moshi` reference. Two execution modes are provided:
//!
//! - Non-streaming (whole-file): [`mimi::Mimi::encode`] / [`mimi::Mimi::decode`]
//!   build a single computation over the whole clip.
//! - Streaming: [`mimi::Mimi::encode_step`] / [`mimi::Mimi::decode_step`] build a
//!   static per-frame step computation, taking the running state plus one frame
//!   and returning the next state plus one slice of codes (or one audio frame).
pub mod conv;
pub mod mimi;
pub mod quantization;
pub mod seanet;
pub mod transformer;

pub use xla_nn::{Error, Result};

use xla::{ElementType, XlaBuilder, XlaOp};

/// Threads streaming state through a `*_step` computation.
///
/// State tensors are extra parameters of the step computation, appended after
/// the input frame and the model weights. Each streaming module declares its
/// input-state parameters via [`StepCtx::state_in`] (recording their shape) and
/// writes the corresponding updated state via [`StepCtx::state_out`]. The order
/// of the collected new-state outputs matches the order of the declared input
/// parameters, so at runtime the outputs can be fed straight back in as the next
/// step's state.
pub struct StepCtx<'a> {
    builder: &'a XlaBuilder,
    next_param: i64,
    state_shapes: Vec<(ElementType, Vec<i64>)>,
    new_states: Vec<Option<XlaOp>>,
}

impl<'a> StepCtx<'a> {
    /// `first_state_param` is the parameter index of the first state tensor,
    /// i.e. right after the input frame and all the model weights.
    pub fn new(builder: &'a XlaBuilder, first_state_param: i64) -> Self {
        Self {
            builder,
            next_param: first_state_param,
            state_shapes: Vec::new(),
            new_states: Vec::new(),
        }
    }

    /// Declare an input-state parameter, returning its slot index and node.
    pub fn state_in(&mut self, ty: ElementType, dims: &[i64]) -> Result<(usize, XlaOp)> {
        let idx = self.state_shapes.len();
        let p = self.builder.parameter(self.next_param, ty, dims, &format!("state{idx}"))?;
        self.next_param += 1;
        self.state_shapes.push((ty, dims.to_vec()));
        self.new_states.push(None);
        Ok((idx, p))
    }

    /// Record the updated value for the state slot returned by [`state_in`](Self::state_in).
    pub fn state_out(&mut self, idx: usize, x: XlaOp) {
        self.new_states[idx] = Some(x);
    }

    /// The (dtype, shape) of every state tensor, in parameter order. Callers use
    /// this to allocate the zero-initialised state buffers.
    pub fn state_shapes(&self) -> &[(ElementType, Vec<i64>)] {
        &self.state_shapes
    }

    /// The updated state nodes, in parameter order (each slot must have been set).
    pub fn into_new_states(self) -> Vec<XlaOp> {
        self.new_states.into_iter().map(|o| o.expect("a state slot was never written")).collect()
    }
}

/// A thin helper that mirrors the `Path` used in the `xn` reference: it keeps a
/// dotted prefix and forwards weight declarations to an [`xla_nn::VarBuilder`],
/// so the safetensors names line up exactly with the reference implementation.
#[derive(Clone)]
pub struct Vb<'a> {
    inner: &'a xla_nn::VarBuilder,
    prefix: String,
}

impl<'a> Vb<'a> {
    pub fn new(inner: &'a xla_nn::VarBuilder) -> Self {
        Self { inner, prefix: String::new() }
    }

    /// Push a component onto the prefix (like `cd`-ing into a directory).
    pub fn pp(&self, s: impl std::fmt::Display) -> Vb<'a> {
        let prefix =
            if self.prefix.is_empty() { s.to_string() } else { format!("{}.{s}", self.prefix) };
        Vb { inner: self.inner, prefix }
    }

    fn key(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }

    /// Declare a weight parameter and return the corresponding graph node.
    pub fn var(&self, name: &str, dims: &[i64]) -> Result<XlaOp> {
        self.inner.var(&self.key(name), dims)
    }

    pub fn var_builder(&self) -> &'a xla_nn::VarBuilder {
        self.inner
    }
}

/// Broadcast-add a per-channel bias of shape `[c]` onto a `[b, c, t]` tensor.
pub(crate) fn add_bias(xs: &XlaOp, bias: &XlaOp) -> Result<XlaOp> {
    let c = bias.dims()?[0] as i64;
    let bias = bias.reshape(&[1, c, 1])?;
    Ok(xs.add_(&bias)?)
}
