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
pub mod lm;
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
    first_state_param: i64,
    next_param: i64,
    state_shapes: Vec<(ElementType, Vec<i64>)>,
    state_ins: Vec<XlaOp>,
    new_states: Vec<Option<XlaOp>>,
    is_first: Option<XlaOp>,
    mask: Option<XlaOp>,
    reset: Option<XlaOp>,
}

impl<'a> StepCtx<'a> {
    /// `first_state_param` is the parameter index of the first state tensor,
    /// i.e. right after the input frame and all the model weights.
    pub fn new(builder: &'a XlaBuilder, first_state_param: i64) -> Self {
        Self {
            builder,
            first_state_param,
            next_param: first_state_param,
            state_shapes: Vec::new(),
            state_ins: Vec::new(),
            new_states: Vec::new(),
            is_first: None,
            mask: None,
            reset: None,
        }
    }

    /// Provide a per-session activity mask, `[batch]` (s32, non-zero =
    /// active). Inactive sessions do not make progress: their states keep
    /// their previous values ([`state_out`](Self::state_out) selects between
    /// the updated and the previous state per batch element) and the
    /// transformer positions do not advance. All state tensors must then have
    /// the batch as their leading dimension. Without a mask every session
    /// steps unconditionally.
    pub fn set_mask(&mut self, mask: XlaOp) {
        self.mask = Some(mask);
    }

    /// The activity mask, if one was provided.
    pub fn mask(&self) -> Option<&XlaOp> {
        self.mask.as_ref()
    }

    /// The activity mask as a boolean `[batch]` vector, if one was provided.
    pub(crate) fn mask_pred(&self) -> Result<Option<XlaOp>> {
        match &self.mask {
            None => Ok(None),
            Some(m) => {
                let b = m.dims()?[0] as i64;
                Ok(Some(m.gt(&self.builder.c0(0i32)?.broadcast(&[b])?)?))
            }
        }
    }

    /// Register input/output aliases for every declared state tensor: state
    /// `i` (element `output_offset + i` of the root tuple) aliases its input
    /// parameter, so that state updates happen in place rather than into
    /// freshly allocated buffers. The state input buffers are donated at
    /// execution time and must not be used after the step; feeding the step's
    /// output states as the next step's inputs (as the streaming loops do) is
    /// exactly that pattern.
    pub fn setup_aliases(&self, output_offset: usize) {
        for i in 0..self.state_shapes.len() {
            self.builder.setup_alias((output_offset + i) as i64, self.first_state_param + i as i64);
        }
    }

    /// Provide an `is_first` flag: non-zero on a session's first step, zero
    /// afterwards, either as a scalar shared by the whole batch or as a
    /// per-session `[batch]` vector. `Replicate`-padded convs use it to
    /// reproduce the whole-file left padding exactly on the first step
    /// (without it they fall back to zero left padding), and the ASR LM uses
    /// it to substitute the audio padding token and the text start token.
    pub fn set_is_first(&mut self, is_first: XlaOp) {
        self.is_first = Some(is_first);
    }

    /// The `is_first` flag as a boolean scalar or `[batch]` vector, if one
    /// was provided.
    pub(crate) fn is_first_pred(&self) -> Result<Option<XlaOp>> {
        match &self.is_first {
            None => Ok(None),
            Some(f) => {
                let dims: Vec<i64> = f.dims()?.iter().map(|d| *d as i64).collect();
                let zeros = self.builder.c0(0i32)?.broadcast(&dims)?;
                Ok(Some(f.gt(&zeros)?))
            }
        }
    }

    /// Provide a per-session reset vector, `[batch]` (s32, non-zero = reset).
    /// All the states of a reset session are zeroed at the start of the step,
    /// so the slot behaves like a fresh stream; combine with a per-session
    /// `is_first` for that step. Resets apply regardless of the activity
    /// mask. Must be set before the model declares its states.
    pub fn set_reset(&mut self, reset: XlaOp) {
        self.reset = Some(reset);
    }

    /// The reset vector, if one was provided.
    pub fn reset(&self) -> Option<&XlaOp> {
        self.reset.as_ref()
    }

    /// Declare an input-state parameter, returning its slot index and node.
    /// When a reset vector is set, the state of reset sessions reads as zero
    /// (the state's leading dimension must then be the batch); the zeroed
    /// value is also what inactive sessions keep through
    /// [`state_out`](Self::state_out), so a reset sticks even on a masked
    /// step.
    pub fn state_in(&mut self, ty: ElementType, dims: &[i64]) -> Result<(usize, XlaOp)> {
        let idx = self.state_shapes.len();
        let p = self.builder.parameter(self.next_param, ty, dims, &format!("state{idx}"))?;
        self.next_param += 1;
        let p = match &self.reset {
            None => p,
            Some(reset) => {
                let b = reset.dims()?[0] as i64;
                let pred = reset.gt(&self.builder.c0(0i32)?.broadcast(&[b])?)?;
                let zero = self.builder.zero(ty)?.broadcast(dims)?;
                pred.broadcast_in_dim(dims, &[0])?.select(&zero, &p)?
            }
        };
        self.state_shapes.push((ty, dims.to_vec()));
        self.state_ins.push(p.clone());
        self.new_states.push(None);
        Ok((idx, p))
    }

    /// Like [`state_in`](Self::state_in) but exempt from the reset zeroing,
    /// for states whose stale contents are unreachable after a reset anyway.
    /// The kv caches are the important case: their validity masks derive from
    /// the position state, so resetting the position alone hides every stale
    /// entry (each cache slot is rewritten before it becomes visible again),
    /// and skipping the zeroing keeps the update free of a full-cache select
    /// that would break the in-place aliasing.
    pub fn state_in_no_reset(&mut self, ty: ElementType, dims: &[i64]) -> Result<(usize, XlaOp)> {
        let idx = self.state_shapes.len();
        let p = self.builder.parameter(self.next_param, ty, dims, &format!("state{idx}"))?;
        self.next_param += 1;
        self.state_shapes.push((ty, dims.to_vec()));
        self.state_ins.push(p.clone());
        self.new_states.push(None);
        Ok((idx, p))
    }

    /// Record the updated value for the state slot returned by
    /// [`state_in`](Self::state_in). When an activity mask is set, inactive
    /// batch elements keep their previous state instead (the state's leading
    /// dimension must be the batch).
    pub fn state_out(&mut self, idx: usize, x: XlaOp) -> Result<()> {
        let x = match self.mask_pred()? {
            None => x,
            Some(pred) => {
                let old = &self.state_ins[idx];
                let dims: Vec<i64> = old.dims()?.iter().map(|d| *d as i64).collect();
                pred.broadcast_in_dim(&dims, &[0])?.select(&x, old)?
            }
        };
        self.new_states[idx] = Some(x);
        Ok(())
    }

    /// Record the updated value for a state slot without applying the
    /// activity mask, for states whose update handles inactive sessions
    /// itself (e.g. the ring kv cache writes and the position counters).
    pub fn state_out_raw(&mut self, idx: usize, x: XlaOp) {
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
