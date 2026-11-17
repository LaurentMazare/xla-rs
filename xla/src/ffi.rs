//! Custom calls through the XLA FFI (`API_VERSION_TYPED_FFI`).
//!
//! Unlike the original custom-call api ([`crate::register_custom_call_target`]),
//! an FFI handler can be marked as safe to capture in command buffers (CUDA
//! graphs), so a custom call no longer splits the graphs XLA builds around it.
//!
//! A handler implements [`Handler`] and is registered with
//! [`register_handler`] under a target name, which ops built with
//! [`XlaBuilder::custom_call_ffi`](crate::XlaBuilder::custom_call_ffi) call.
//! It reads its operand and result buffers, named attributes and stream from
//! the [`CallFrame`].
//!
//! ```ignore
//! struct Axpy;
//! impl xla::ffi::Handler for Axpy {
//!     const TRAITS: HandlerTraits = HandlerTraits::COMMAND_BUFFER_COMPATIBLE;
//!     fn execute(frame: &CallFrame) -> Result<(), String> {
//!         let a: f32 = frame.attr("a").ok_or("missing attribute a")?;
//!         let (x, y, out) = (frame.arg(0), frame.arg(1), frame.ret(0));
//!         // ... launch on frame.stream() (CUDA) or compute on the host ...
//!         Ok(())
//!     }
//! }
//! xla::ffi::register_handler::<Axpy>("axpy", "CUDA")?;
//! let out = builder.custom_call_ffi("axpy", &[x, y], ElementType::F32, &[n], "{a = 2.0 : f32}")?;
//! ```
//!
//! The structs mirror `xla/ffi/api/c_api.h`, restricted to the parts that are
//! the same across the XLA versions xla-rs supports (FFI api 0.2 to 0.4).
use std::ffi::{c_char, c_int, c_void, CString};

use crate::{Error as XlaError, Result};

/// `XLA_FFI_Error`, owned by whoever receives it.
#[repr(C)]
struct Error {
    _private: [u8; 0],
}

/// `XLA_FFI_Handler_Traits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HandlerTraits(pub u32);

impl HandlerTraits {
    pub const NONE: Self = Self(0);
    /// Every call launches the same device work (it may depend on the
    /// attributes), so XLA may record it once in a command buffer (a CUDA
    /// graph) and replay it: buffers can change address between replays, so
    /// the handler must only enqueue work on [`CallFrame::stream`], with no
    /// host-side effects.
    pub const COMMAND_BUFFER_COMPATIBLE: Self = Self(1);
}

/// An FFI custom call target, see the [module docs](self).
pub trait Handler {
    /// What the handler tells XLA it may do with its calls.
    const TRAITS: HandlerTraits = HandlerTraits::NONE;

    /// Runs one call; an error fails the execution with its message.
    fn execute(frame: &CallFrame) -> std::result::Result<(), String>;
}

/// The C entry point XLA calls for `H`: answers XLA's metadata query (the
/// handler's api version and traits, asked at registration), checks the
/// stage, and turns errors and panics into XLA errors.
unsafe extern "C" fn entry<H: Handler>(frame: *mut CallFrame) -> *mut Error {
    let frame = unsafe { &*frame };
    if !frame.extension_start.is_null() {
        let ext = frame.extension_start as *const MetadataExtension;
        if unsafe { (*ext).base.ty } == EXTENSION_METADATA {
            let api = unsafe { &(*frame.api).api_version };
            if api.major_version != 0 {
                return frame.error(&format!("unsupported FFI api {}", api.major_version));
            }
            let metadata = unsafe { &mut *(*ext).metadata };
            metadata.api_version = ApiVersion {
                struct_size: std::mem::size_of::<ApiVersion>(),
                extension_start: std::ptr::null_mut(),
                major_version: api.major_version,
                minor_version: api.minor_version,
            };
            metadata.traits = H::TRAITS.0;
            return std::ptr::null_mut();
        }
    }
    if frame.stage != ExecutionStage::EXECUTE {
        return frame.error(&format!("unexpected execution stage {:?}", frame.stage));
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| H::execute(frame))) {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(msg)) => frame.error(&msg),
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            frame.error(&format!("ffi handler panicked: {msg}"))
        }
    }
}

/// Register `H` as the FFI custom call target `name` for `platform`
/// (`"CUDA"`, `"Host"` for the cpu client, ...). Registering the same handler
/// again is a no-op, another handler under a taken name is an error.
pub fn register_handler<H: Handler>(name: &str, platform: &str) -> Result<()> {
    let name = CString::new(name)?;
    let platform = CString::new(platform)?;
    let status = unsafe {
        crate::c_lib::register_ffi_handler(
            name.as_ptr(),
            platform.as_ptr(),
            entry::<H> as *mut c_void,
            H::TRAITS.0,
        )
    };
    crate::wrappers::handle_status(status)
}

/// `XLA_FFI_DataType`, the element type of a buffer or scalar attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct DataType(pub c_int);

impl DataType {
    pub const PRED: Self = Self(1);
    pub const S8: Self = Self(2);
    pub const S16: Self = Self(3);
    pub const S32: Self = Self(4);
    pub const S64: Self = Self(5);
    pub const U8: Self = Self(6);
    pub const U16: Self = Self(7);
    pub const U32: Self = Self(8);
    pub const U64: Self = Self(9);
    pub const F16: Self = Self(10);
    pub const F32: Self = Self(11);
    pub const F64: Self = Self(12);
    pub const BF16: Self = Self(16);
}

/// `XLA_FFI_ExecutionStage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ExecutionStage(pub c_int);

impl ExecutionStage {
    pub const INSTANTIATE: Self = Self(0);
    pub const PREPARE: Self = Self(1);
    pub const INITIALIZE: Self = Self(2);
    pub const EXECUTE: Self = Self(3);
}

#[repr(C)]
struct ByteSpan {
    ptr: *const c_char,
    len: usize,
}

impl ByteSpan {
    fn as_bytes(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
}

/// An operand or result of the call: device memory on GPU, host memory on
/// cpu.
#[repr(C)]
pub struct Buffer {
    struct_size: usize,
    extension_start: *mut c_void,
    dtype: DataType,
    data: *mut c_void,
    rank: i64,
    dims: *mut i64,
}

impl Buffer {
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn data(&self) -> *mut c_void {
        self.data
    }

    pub fn dims(&self) -> &[i64] {
        if self.rank == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.dims, self.rank as usize) }
    }

    pub fn element_count(&self) -> usize {
        self.dims().iter().product::<i64>() as usize
    }
}

#[repr(C)]
struct Args {
    struct_size: usize,
    extension_start: *mut c_void,
    size: i64,
    types: *mut c_int,
    args: *mut *mut c_void,
}

#[repr(C)]
struct Attrs {
    struct_size: usize,
    extension_start: *mut c_void,
    size: i64,
    types: *mut c_int,
    names: *mut *mut ByteSpan,
    attrs: *mut *mut c_void,
}

const ARG_BUFFER: c_int = 1;
const ATTR_SCALAR: c_int = 3;
const ATTR_STRING: c_int = 4;

#[repr(C)]
struct Scalar {
    dtype: DataType,
    value: *mut c_void,
}

/// The leading entries of `XLA_FFI_Api`, the same in every supported version.
#[repr(C)]
struct Api {
    struct_size: usize,
    extension_start: *mut c_void,
    api_version: ApiVersion,
    internal_api: *const c_void,
    error_create: unsafe extern "C" fn(*mut ErrorCreateArgs) -> *mut Error,
    error_get_message: *const c_void,
    error_destroy: unsafe extern "C" fn(*mut ErrorDestroyArgs),
    handler_register: *const c_void,
    stream_get: unsafe extern "C" fn(*mut StreamGetArgs) -> *mut Error,
}

#[repr(C)]
struct ApiVersion {
    struct_size: usize,
    extension_start: *mut c_void,
    major_version: c_int,
    minor_version: c_int,
}

#[repr(C)]
struct ErrorCreateArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    message: *const c_char,
    errc: c_int,
}

#[repr(C)]
struct ErrorDestroyArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    error: *mut Error,
}

#[repr(C)]
struct StreamGetArgs {
    struct_size: usize,
    extension_start: *mut c_void,
    ctx: *mut c_void,
    stream: *mut c_void,
}

const EXTENSION_METADATA: c_int = 1;

#[repr(C)]
struct ExtensionBase {
    struct_size: usize,
    ty: c_int,
    next: *mut ExtensionBase,
}

#[repr(C)]
struct MetadataExtension {
    base: ExtensionBase,
    metadata: *mut Metadata,
}

#[repr(C)]
struct Metadata {
    struct_size: usize,
    api_version: ApiVersion,
    traits: u32,
}

/// What XLA passes to a handler: `XLA_FFI_CallFrame`.
#[repr(C)]
pub struct CallFrame {
    struct_size: usize,
    extension_start: *mut ExtensionBase,
    api: *const Api,
    ctx: *mut c_void,
    stage: ExecutionStage,
    args: Args,
    rets: Args,
    attrs: Attrs,
    future: *mut c_void,
}

/// A scalar attribute type, see [`CallFrame::attr`].
pub trait ScalarAttr: Copy {
    const DTYPE: DataType;
}

macro_rules! scalar_attr {
    ($($t:ty => $d:ident),*) => {
        $(impl ScalarAttr for $t { const DTYPE: DataType = DataType::$d; })*
    };
}
scalar_attr!(bool => PRED, i8 => S8, i16 => S16, i32 => S32, i64 => S64, u8 => U8, u16 => U16,
    u32 => U32, u64 => U64, f32 => F32, f64 => F64);

impl CallFrame {
    pub fn stage(&self) -> ExecutionStage {
        self.stage
    }

    pub fn num_args(&self) -> usize {
        self.args.size as usize
    }

    pub fn num_rets(&self) -> usize {
        self.rets.size as usize
    }

    fn buffer(list: &Args, i: usize) -> Option<&Buffer> {
        if i >= list.size as usize {
            return None;
        }
        unsafe {
            if *list.types.add(i) != ARG_BUFFER {
                return None;
            }
            Some(&*(*list.args.add(i) as *const Buffer))
        }
    }

    /// Operand `i`.
    pub fn arg(&self, i: usize) -> Option<&Buffer> {
        Self::buffer(&self.args, i)
    }

    /// Result `i`.
    pub fn ret(&self, i: usize) -> Option<&Buffer> {
        Self::buffer(&self.rets, i)
    }

    fn find_attr(&self, name: &str) -> Option<(c_int, *mut c_void)> {
        (0..self.attrs.size as usize).find_map(|i| unsafe {
            let n = &**self.attrs.names.add(i);
            (n.as_bytes() == name.as_bytes())
                .then(|| (*self.attrs.types.add(i), *self.attrs.attrs.add(i)))
        })
    }

    /// The scalar attribute `name`, `None` when missing or of another type.
    pub fn attr<T: ScalarAttr>(&self, name: &str) -> Option<T> {
        let (ty, ptr) = self.find_attr(name)?;
        if ty != ATTR_SCALAR {
            return None;
        }
        let s = unsafe { &*(ptr as *const Scalar) };
        (s.dtype == T::DTYPE).then(|| unsafe { std::ptr::read_unaligned(s.value as *const T) })
    }

    /// The string attribute `name`.
    pub fn attr_str(&self, name: &str) -> Option<&[u8]> {
        let (ty, ptr) = self.find_attr(name)?;
        (ty == ATTR_STRING).then(|| unsafe { (*(ptr as *const ByteSpan)).as_bytes() })
    }

    /// The platform stream the call runs on: a `CUstream` on CUDA.
    pub fn stream(&self) -> Result<*mut c_void> {
        let mut args = StreamGetArgs {
            struct_size: std::mem::size_of::<StreamGetArgs>(),
            extension_start: std::ptr::null_mut(),
            ctx: self.ctx,
            stream: std::ptr::null_mut(),
        };
        let err = unsafe { ((*self.api).stream_get)(&mut args) };
        if !err.is_null() {
            let mut destroy = ErrorDestroyArgs {
                struct_size: std::mem::size_of::<ErrorDestroyArgs>(),
                extension_start: std::ptr::null_mut(),
                error: err,
            };
            unsafe { ((*self.api).error_destroy)(&mut destroy) };
            return Err(XlaError::XlaError {
                msg: "XLA_FFI_Stream_Get failed".to_string(),
                backtrace: String::new(),
            });
        }
        Ok(args.stream)
    }

    /// An internal error with `msg`, for the entry point to return.
    fn error(&self, msg: &str) -> *mut Error {
        let msg = CString::new(msg.replace('\0', " ")).unwrap();
        let mut args = ErrorCreateArgs {
            struct_size: std::mem::offset_of!(ErrorCreateArgs, errc) + std::mem::size_of::<c_int>(),
            extension_start: std::ptr::null_mut(),
            message: msg.as_ptr(),
            errc: 13, // XLA_FFI_Error_Code_INTERNAL
        };
        unsafe { ((*self.api).error_create)(&mut args) }
    }
}
