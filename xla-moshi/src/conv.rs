//! Streamable 1D convolutions, ported from the `xn-moshi` reference but running
//! over the whole sequence at once (non-streaming). Only the forward path is
//! implemented.
//!
//! Weight-norm is assumed to be pre-fused into a single `weight` tensor, which
//! is the case for the candle-format Mimi checkpoints used by the AudioToAudio
//! example.
use crate::{add_bias, Result, StepCtx, Vb};
use xla::{ElementType, XlaOp};

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Norm {
    WeightNorm,
    SpectralNorm,
    TimeGroupNorm,
    None,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PadMode {
    Constant,
    Reflect,
    Replicate,
}

fn get_extra_padding_for_conv1d(len: i64, k_size_eff: i64, stride: i64, padding_total: i64) -> i64 {
    let n_frames = (len + padding_total - k_size_eff).max(0) as f64 / stride as f64 + 1.0;
    let ideal_len = ((n_frames.ceil() as i64 - 1) * stride + k_size_eff - padding_total).max(0);
    (ideal_len - len).max(0)
}

fn pad1d(xs: &XlaOp, pad_l: i64, pad_r: i64, mode: PadMode) -> Result<XlaOp> {
    match mode {
        PadMode::Constant => {
            let zero = xs.builder().c0(0f32)?;
            Ok(xs.pad_in_dim(&zero, 2, pad_l, pad_r)?)
        }
        PadMode::Replicate => pad_replicate(xs, pad_l, pad_r),
        PadMode::Reflect => Err(crate::Error::Xla(xla::Error::XlaError {
            msg: "pad-mode 'reflect' is not supported".to_string(),
            backtrace: String::new(),
        })),
    }
}

/// Edge (replicate) padding along the time dimension.
fn pad_replicate(xs: &XlaOp, pad_l: i64, pad_r: i64) -> Result<XlaOp> {
    let dims = xs.dims()?;
    let (b, c, len) = (dims[0] as i64, dims[1] as i64, dims[2] as i64);
    let mut parts: Vec<XlaOp> = Vec::new();
    if pad_l > 0 {
        let edge = xs.slice_in_dim1(0, 1, 2)?;
        parts.push(edge.broadcast_in_dim(&[b, c, pad_l], &[0, 1, 2])?);
    }
    parts.push(xs.clone());
    if pad_r > 0 {
        let edge = xs.slice_in_dim1(len - 1, len, 2)?;
        parts.push(edge.broadcast_in_dim(&[b, c, pad_r], &[0, 1, 2])?);
    }
    if parts.len() == 1 {
        return Ok(parts.pop().unwrap());
    }
    let first = parts.remove(0);
    Ok(first.concat_in_dim(&parts, 2)?)
}

fn unpad1d(xs: &XlaOp, unpad_l: i64, unpad_r: i64) -> Result<XlaOp> {
    let len = xs.dims()?[2] as i64;
    Ok(xs.slice_in_dim1(unpad_l, len - unpad_r, 2)?)
}

pub struct StreamableConv1d {
    weight: XlaOp,
    bias: Option<XlaOp>,
    stride: i64,
    dilation: i64,
    groups: i64,
    kernel_size: i64,
    causal: bool,
    pad_mode: PadMode,
}

impl StreamableConv1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Vb,
        in_c: i64,
        out_c: i64,
        k_size: i64,
        stride: i64,
        dilation: i64,
        groups: i64,
        bias: bool,
        causal: bool,
        _norm: Option<Norm>,
        pad_mode: PadMode,
    ) -> Result<Self> {
        let vb = vb.pp("conv").pp("conv");
        let weight = vb.var("weight", &[out_c, in_c / groups, k_size])?;
        let bias = if bias { Some(vb.var("bias", &[out_c])?) } else { None };
        Ok(Self { weight, bias, stride, dilation, groups, kernel_size: k_size, causal, pad_mode })
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let k_size_eff = (self.kernel_size - 1) * self.dilation + 1;
        let padding_total = k_size_eff - self.stride;
        let len = xs.dims()?[2] as i64;
        let extra = get_extra_padding_for_conv1d(len, k_size_eff, self.stride, padding_total);
        let xs = if self.causal {
            pad1d(xs, padding_total, extra, self.pad_mode)?
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            pad1d(xs, padding_left, padding_right + extra, self.pad_mode)?
        };
        let ys = xs.conv1d(&self.weight, self.stride, 0, self.dilation, self.groups)?;
        match &self.bias {
            Some(b) => add_bias(&ys, b),
            None => Ok(ys),
        }
    }

    /// The number of input samples carried between streaming steps (the causal
    /// left context of the sliding window).
    fn carry(&self) -> i64 {
        let k_size_eff = (self.kernel_size - 1) * self.dilation + 1;
        k_size_eff - self.stride
    }

    /// Streaming step: `xs` is `[1, in_c, l]` with `l` a multiple of `stride`;
    /// returns `[1, out_c, l / stride]`. The carried input history is threaded
    /// through `ctx` (a zero-initialised buffer of shape `[1, in_c, carry]`,
    /// which also supplies the causal left padding on the first steps).
    ///
    /// For `PadMode::Replicate` convs the whole-file forward instead replicates
    /// the first frame into the left pad. Pass an `is_first` flag through `ctx`
    /// (see [`StepCtx::set_is_first`](crate::StepCtx::set_is_first)) and this
    /// step reproduces that exactly on the first step; without it the zero left
    /// padding causes a one-frame warm-up difference.
    pub fn step(&self, xs: &XlaOp, ctx: &mut StepCtx) -> Result<XlaOp> {
        let carry = self.carry();
        let wd = self.weight.dims()?;
        let in_c = wd[1] as i64 * self.groups;
        let b = xs.dims()?[0] as i64;
        let combined = if carry > 0 {
            let is_first = ctx.is_first_pred()?;
            let (idx, state) = ctx.state_in(ElementType::F32, &[b, in_c, carry])?;
            // The left pad is normally the carried input history (which is
            // zero-initialised, matching `Constant` padding). For `Replicate`
            // padding the whole-file forward replicates the first input column
            // instead, so on the first step select that replicated pad.
            let pad = match (self.pad_mode, is_first) {
                (PadMode::Replicate, Some(is_first)) => {
                    let repl = xs
                        .slice_in_dim1(0, 1, 2)?
                        .broadcast_in_dim(&[b, in_c, carry], &[0, 1, 2])?;
                    // The flag is either a scalar or a per-session [b] vector.
                    let is_first = if is_first.rank()? == 0 {
                        is_first.broadcast(&[b, in_c, carry])?
                    } else {
                        is_first.broadcast_in_dim(&[b, in_c, carry], &[0])?
                    };
                    is_first.select(&repl, &state)?
                }
                _ => state,
            };
            let combined = pad.concat_in_dim(std::slice::from_ref(xs), 2)?;
            let clen = combined.dims()?[2] as i64;
            ctx.state_out(idx, combined.slice_in_dim1(clen - carry, clen, 2)?)?;
            combined
        } else {
            xs.clone()
        };
        let ys = combined.conv1d(&self.weight, self.stride, 0, self.dilation, self.groups)?;
        match &self.bias {
            Some(b) => add_bias(&ys, b),
            None => Ok(ys),
        }
    }
}

pub struct StreamableConvTranspose1d {
    weight: XlaOp,
    bias: Option<XlaOp>,
    k_size: i64,
    stride: i64,
    groups: i64,
    causal: bool,
}

impl StreamableConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Vb,
        in_c: i64,
        out_c: i64,
        k_size: i64,
        stride: i64,
        groups: i64,
        bias: bool,
        causal: bool,
        _norm: Option<Norm>,
    ) -> Result<Self> {
        let vb = vb.pp("convtr").pp("convtr");
        let weight = vb.var("weight", &[in_c, out_c / groups, k_size])?;
        let bias = if bias { Some(vb.var("bias", &[out_c])?) } else { None };
        Ok(Self { weight, bias, k_size, stride, groups, causal })
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        let padding_total = (self.k_size - self.stride).max(0);
        let ys = xs.conv_transpose1d(&self.weight, self.stride, 0, 0, 1, self.groups)?;
        let ys = match &self.bias {
            Some(b) => add_bias(&ys, b)?,
            None => ys,
        };
        if self.causal {
            unpad1d(&ys, 0, padding_total)
        } else {
            let padding_right = padding_total / 2;
            let padding_left = padding_total - padding_right;
            unpad1d(&ys, padding_left, padding_right)
        }
    }

    /// Streaming step for the causal transposed conv: `xs` is `[1, in_c, l]`,
    /// returns `[1, out_c, l * stride]`. The overlap-add tail is carried through
    /// `ctx` (a zero-initialised `[1, out_c, k - stride]` buffer). The bias is
    /// applied to the emitted output, and the carried tail is kept bias-free so
    /// it is not double-counted on the next step.
    pub fn step(&self, xs: &XlaOp, ctx: &mut StepCtx) -> Result<XlaOp> {
        let carry = (self.k_size - self.stride).max(0);
        let wd = self.weight.dims()?;
        let out_c = wd[1] as i64 * self.groups;
        let b = xs.dims()?[0] as i64;
        // Raw transposed conv without bias.
        let ys = xs.conv_transpose1d(&self.weight, self.stride, 0, 0, 1, self.groups)?;
        let ot = ys.dims()?[2] as i64;
        let ys = if carry > 0 {
            let (idx, state) = ctx.state_in(ElementType::F32, &[b, out_c, carry])?;
            // Overlap-add the carried tail onto the head of this step's output.
            let head = ys.slice_in_dim1(0, carry, 2)?.add_(&state)?;
            let tail = ys.slice_in_dim1(carry, ot, 2)?;
            let ys = head.concat_in_dim(&[tail], 2)?;
            let valid_len = ot - carry;
            ctx.state_out(idx, ys.slice_in_dim1(valid_len, ot, 2)?)?;
            ys.slice_in_dim1(0, valid_len, 2)?
        } else {
            ys
        };
        match &self.bias {
            Some(b) => add_bias(&ys, b),
            None => Ok(ys),
        }
    }
}

pub struct ConvDownsample1d {
    conv: StreamableConv1d,
}

impl ConvDownsample1d {
    pub fn load(vb: &Vb, stride: i64, dim: i64, causal: bool) -> Result<Self> {
        let conv = StreamableConv1d::load(
            &vb.pp("conv"),
            dim,
            dim,
            2 * stride,
            stride,
            1,
            1,
            false,
            causal,
            None,
            PadMode::Replicate,
        )?;
        Ok(Self { conv })
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        self.conv.forward(xs)
    }

    pub fn step(&self, xs: &XlaOp, ctx: &mut StepCtx) -> Result<XlaOp> {
        self.conv.step(xs, ctx)
    }
}

pub struct ConvTrUpsample1d {
    convtr: StreamableConvTranspose1d,
}

impl ConvTrUpsample1d {
    pub fn load(vb: &Vb, stride: i64, dim: i64, causal: bool) -> Result<Self> {
        let convtr = StreamableConvTranspose1d::load(
            &vb.pp("convtr"),
            dim,
            dim,
            2 * stride,
            stride,
            dim, // depthwise
            false,
            causal,
            None,
        )?;
        Ok(Self { convtr })
    }

    pub fn forward(&self, xs: &XlaOp) -> Result<XlaOp> {
        self.convtr.forward(xs)
    }

    pub fn step(&self, xs: &XlaOp, ctx: &mut StepCtx) -> Result<XlaOp> {
        self.convtr.step(xs, ctx)
    }
}
