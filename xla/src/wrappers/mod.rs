mod literal;
mod pjrt_buffer;
mod pjrt_client;
mod pjrt_device;
mod pjrt_loaded_executable;
mod shape;
mod xla_builder;
mod xla_op;

use crate::c_lib;
use crate::error::{Error, Result};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

pub use literal::Literal;
pub use pjrt_buffer::PjRtBuffer;
pub use pjrt_client::{PjRtClient, PluginOption};
pub use pjrt_device::PjRtDevice;
pub use pjrt_loaded_executable::PjRtLoadedExecutable;
pub use shape::{ArrayShape, Shape};
pub use xla_builder::XlaBuilder;
pub use xla_op::XlaOp;

unsafe fn c_ptr_to_string(ptr: *const std::ffi::c_char) -> String {
    let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
    libc::free(ptr as *mut libc::c_void);
    str
}

/// The primitive types supported by XLA. `S8` is a signed 1 byte integer,
/// `U32` is an unsigned 4 bytes integer, etc.
#[derive(Clone, Copy, PartialEq, Eq, Debug, FromPrimitive)]
pub enum PrimitiveType {
    Invalid = 0,
    Pred = 1,
    S8 = 2,
    S16 = 3,
    S32 = 4,
    S64 = 5,
    U8 = 6,
    U16 = 7,
    U32 = 8,
    U64 = 9,
    F16 = 10,
    F32 = 11,
    Bf16 = 16,
    F64 = 12,
    /// 8-bit float with a 5-bit exponent and a 2-bit mantissa.
    F8E5M2 = 19,
    /// 8-bit float with a 4-bit exponent and a 3-bit mantissa, finite-only
    /// (no infinities, a single NaN bit-pattern).
    F8E4M3FN = 20,
    /// 4-bit float with a 2-bit exponent and a 1-bit mantissa, finite-only.
    /// This is the MX (microscaling) element type; note that it is sub-byte:
    /// XLA packs two elements per byte in literals and device buffers, so the
    /// byte-oriented host-transfer helpers do not support it.
    F4E2M1FN = 32,
    /// 8-bit unsigned float with an 8-bit exponent, no mantissa and no sign:
    /// the power-of-two MX (microscaling) shared-scale type. It has no zero,
    /// no infinities and a single NaN bit-pattern.
    F8E8M0FNU = 33,
    C64 = 15,
    C128 = 18,
    Tuple = 13,
    OpaqueType = 14,
    Token = 17,
}

impl PrimitiveType {
    /// The [`ElementType`] for this primitive type, or an error for the
    /// non-element types (`Invalid`, `Tuple`, `OpaqueType`, `Token`).
    pub fn element_type(self) -> Result<ElementType> {
        match self {
            Self::Pred => Ok(ElementType::Pred),
            Self::S8 => Ok(ElementType::S8),
            Self::S16 => Ok(ElementType::S16),
            Self::S32 => Ok(ElementType::S32),
            Self::S64 => Ok(ElementType::S64),
            Self::U8 => Ok(ElementType::U8),
            Self::U16 => Ok(ElementType::U16),
            Self::U32 => Ok(ElementType::U32),
            Self::U64 => Ok(ElementType::U64),
            Self::F16 => Ok(ElementType::F16),
            Self::F32 => Ok(ElementType::F32),
            Self::Bf16 => Ok(ElementType::Bf16),
            Self::F64 => Ok(ElementType::F64),
            Self::F8E5M2 => Ok(ElementType::F8E5M2),
            Self::F8E4M3FN => Ok(ElementType::F8E4M3FN),
            Self::F4E2M1FN => Ok(ElementType::F4E2M1FN),
            Self::F8E8M0FNU => Ok(ElementType::F8E8M0FNU),
            Self::C64 => Ok(ElementType::C64),
            Self::C128 => Ok(ElementType::C128),
            Self::Invalid | Self::Tuple | Self::OpaqueType | Self::Token => {
                Err(Error::NotAnElementType { got: self })
            }
        }
    }
}

/// The algorithm used by [`XlaOp::rng_bit_generator`](crate::XlaOp::rng_bit_generator).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RandomAlgorithm {
    /// The backend picks its preferred algorithm.
    Default = 0,
    ThreeFry = 1,
    Philox = 2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ElementType {
    Pred,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    Bf16,
    F64,
    F8E5M2,
    F8E4M3FN,
    F4E2M1FN,
    F8E8M0FNU,
    C64,
    C128,
}

impl ElementType {
    /// The size for this element type in bytes.
    pub fn element_size_in_bytes(&self) -> usize {
        match self {
            Self::Pred => 1,
            Self::S8 => 1,
            Self::S16 => 2,
            Self::S32 => 4,
            Self::S64 => 8,
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::Bf16 => 2,
            Self::F64 => 8,
            Self::F8E5M2 => 1,
            Self::F8E4M3FN => 1,
            // Sub-byte: a single unpacked element fits in one byte, but XLA
            // stores f4 data packed two elements per byte — sizes derived
            // from this value must not be used for packed host data (see
            // `PjRtClient::buffer_from_host_raw_bytes`).
            Self::F4E2M1FN => 1,
            Self::F8E8M0FNU => 1,
            Self::C64 => 8,
            Self::C128 => 16,
        }
    }

    pub fn primitive_type(&self) -> PrimitiveType {
        match self {
            Self::Pred => PrimitiveType::Pred,
            Self::S8 => PrimitiveType::S8,
            Self::S16 => PrimitiveType::S16,
            Self::S32 => PrimitiveType::S32,
            Self::S64 => PrimitiveType::S64,
            Self::U8 => PrimitiveType::U8,
            Self::U16 => PrimitiveType::U16,
            Self::U32 => PrimitiveType::U32,
            Self::U64 => PrimitiveType::U64,
            Self::F16 => PrimitiveType::F16,
            Self::F32 => PrimitiveType::F32,
            Self::Bf16 => PrimitiveType::Bf16,
            Self::F64 => PrimitiveType::F64,
            Self::F8E5M2 => PrimitiveType::F8E5M2,
            Self::F8E4M3FN => PrimitiveType::F8E4M3FN,
            Self::F4E2M1FN => PrimitiveType::F4E2M1FN,
            Self::F8E8M0FNU => PrimitiveType::F8E8M0FNU,
            Self::C64 => PrimitiveType::C64,
            Self::C128 => PrimitiveType::C128,
        }
    }
}

pub trait ArrayElement: Copy {
    const TY: ElementType;
    const ELEMENT_SIZE_IN_BYTES: usize;
    const ZERO: Self;
}

#[allow(clippy::missing_safety_doc)]
/// A type implementing the `NativeType` trait can be directly converted to constant ops or
/// literals.
pub trait NativeType: Copy {
    unsafe fn constant_r0(b: c_lib::xla_builder, v: Self) -> c_lib::xla_op;
    unsafe fn constant_r1(b: c_lib::xla_builder, v: *const Self, l: usize) -> c_lib::xla_op;
    unsafe fn constant_r1c(b: c_lib::xla_builder, v: Self, l: usize) -> c_lib::xla_op;
    unsafe fn create_r0(v: Self) -> c_lib::literal;
    unsafe fn create_r1(v: *const Self, l: usize) -> c_lib::literal;
    unsafe fn literal_get_first_element(l: c_lib::literal) -> Self;
}

macro_rules! native_type {
    ($ty:ty, $cst0:ident, $cst1:ident, $cst1c:ident, $cre0:ident, $cre1:ident, $gf:ident) => {
        impl NativeType for $ty {
            unsafe fn constant_r0(b: c_lib::xla_builder, v: Self) -> c_lib::xla_op {
                c_lib::$cst0(b, v)
            }
            unsafe fn constant_r1(
                b: c_lib::xla_builder,
                v: *const Self,
                l: usize,
            ) -> c_lib::xla_op {
                c_lib::$cst1(b, v, l)
            }
            unsafe fn constant_r1c(b: c_lib::xla_builder, v: Self, l: usize) -> c_lib::xla_op {
                c_lib::$cst1c(b, v, l)
            }
            unsafe fn create_r0(v: Self) -> c_lib::literal {
                c_lib::$cre0(v)
            }
            unsafe fn create_r1(v: *const Self, l: usize) -> c_lib::literal {
                c_lib::$cre1(v, l)
            }
            unsafe fn literal_get_first_element(l: c_lib::literal) -> Self {
                c_lib::$gf(l)
            }
        }
    };
}

native_type!(
    i32,
    constant_r0_int32_t,
    constant_r1_int32_t,
    constant_r1c_int32_t,
    create_r0_int32_t,
    create_r1_int32_t,
    literal_get_first_element_int32_t
);

native_type!(
    i64,
    constant_r0_int64_t,
    constant_r1_int64_t,
    constant_r1c_int64_t,
    create_r0_int64_t,
    create_r1_int64_t,
    literal_get_first_element_int64_t
);

native_type!(
    u32,
    constant_r0_uint32_t,
    constant_r1_uint32_t,
    constant_r1c_uint32_t,
    create_r0_uint32_t,
    create_r1_uint32_t,
    literal_get_first_element_uint32_t
);

native_type!(
    u64,
    constant_r0_uint64_t,
    constant_r1_uint64_t,
    constant_r1c_uint64_t,
    create_r0_uint64_t,
    create_r1_uint64_t,
    literal_get_first_element_uint64_t
);

native_type!(
    f32,
    constant_r0_float,
    constant_r1_float,
    constant_r1c_float,
    create_r0_float,
    create_r1_float,
    literal_get_first_element_float
);

native_type!(
    f64,
    constant_r0_double,
    constant_r1_double,
    constant_r1c_double,
    create_r0_double,
    create_r1_double,
    literal_get_first_element_double
);

macro_rules! element_type {
    ($ty:ty, $v:ident, $sz:tt) => {
        impl ArrayElement for $ty {
            const TY: ElementType = ElementType::$v;
            const ELEMENT_SIZE_IN_BYTES: usize = $sz;
            const ZERO: Self = 0 as Self;
        }
    };
}

// Dummy F16 type.
#[derive(Copy, Clone, Debug)]
pub struct F16;

impl ArrayElement for F16 {
    const TY: ElementType = ElementType::F16;
    const ELEMENT_SIZE_IN_BYTES: usize = 2;
    const ZERO: Self = Self;
}

// Dummy BF16 type.
#[derive(Copy, Clone, Debug)]
pub struct Bf16;

impl ArrayElement for Bf16 {
    const TY: ElementType = ElementType::Bf16;
    const ELEMENT_SIZE_IN_BYTES: usize = 2;
    const ZERO: Self = Self;
}

// Dummy F8E5M2 type. Like `F16`/`Bf16` there is no native host representation:
// values are produced and consumed on the host through `convert` to/from a
// wider float type.
#[derive(Copy, Clone, Debug)]
pub struct F8E5M2;

impl ArrayElement for F8E5M2 {
    const TY: ElementType = ElementType::F8E5M2;
    const ELEMENT_SIZE_IN_BYTES: usize = 1;
    const ZERO: Self = Self;
}

// Dummy F8E4M3FN type, see `F8E5M2`.
#[derive(Copy, Clone, Debug)]
pub struct F8E4M3FN;

impl ArrayElement for F8E4M3FN {
    const TY: ElementType = ElementType::F8E4M3FN;
    const ELEMENT_SIZE_IN_BYTES: usize = 1;
    const ZERO: Self = Self;
}

// Dummy F8E8M0FNU type, see `F8E5M2`. There is deliberately no marker type
// for `F4E2M1FN`: it is packed two elements per byte, so none of the
// byte-per-element host paths gated by `ArrayElement` can handle it.
#[derive(Copy, Clone, Debug)]
pub struct F8E8M0FNU;

impl ArrayElement for F8E8M0FNU {
    const TY: ElementType = ElementType::F8E8M0FNU;
    const ELEMENT_SIZE_IN_BYTES: usize = 1;
    const ZERO: Self = Self;
}

element_type!(u8, U8, 1);
element_type!(u16, U16, 2);
element_type!(u32, U32, 4);
element_type!(u64, U64, 8);
element_type!(i8, S8, 1);
element_type!(i16, S16, 2);
element_type!(i32, S32, 4);
element_type!(i64, S64, 8);
element_type!(f32, F32, 4);
element_type!(f64, F64, 8);

/// A computation is built from a root [`XlaOp`]. Computations are device independent and can be
/// specialized to a given device through a compilation step.
pub struct XlaComputation(c_lib::xla_computation);

fn handle_status(status: c_lib::status) -> Result<()> {
    if status.is_null() {
        Ok(())
    } else {
        let msg = unsafe {
            let error_message_ptr = c_lib::status_error_message(status);
            let error_message = c_ptr_to_string(error_message_ptr);
            c_lib::status_free(status);
            error_message
        };
        let backtrace = std::backtrace::Backtrace::capture().to_string();
        Err(Error::XlaError { msg, backtrace })
    }
}

impl XlaComputation {
    pub fn from_proto(proto: &HloModuleProto) -> Self {
        let ptr = unsafe { c_lib::xla_computation_from_hlo_module_proto(proto.0) };
        Self(ptr)
    }

    /// The computation name.
    pub fn name(&self) -> String {
        unsafe {
            let ptr = c_lib::xla_computation_name(self.0);
            c_ptr_to_string(ptr)
        }
    }

    /// Compile this computation for the specified client.
    pub fn compile(&self, client: &PjRtClient) -> Result<PjRtLoadedExecutable> {
        client.compile(self)
    }

    /// Get the HloModuleProto for the computation.
    pub fn proto(&self) -> HloModuleProto {
        let ptr = unsafe { c_lib::xla_computation_proto(self.0) };
        HloModuleProto(ptr)
    }
}

impl Drop for XlaComputation {
    fn drop(&mut self) {
        unsafe { c_lib::xla_computation_free(self.0) }
    }
}

pub struct HloModuleProto(c_lib::hlo_module_proto);

impl HloModuleProto {
    /// Read a HLO module from a text file.
    pub fn from_text_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path.as_ref())?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;
        Self::parse_and_return_unverified_module(&content)
    }

    /// Read a HLO module from a proto file, either in binary or pbtxt format.
    pub fn from_proto_file<P: AsRef<std::path::Path>>(path: P, binary: bool) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path.as_ref())?;
        let mut content = Vec::new();
        file.read_to_end(&mut content)?;
        Self::parse_proto(&content, binary)
    }

    pub fn parse_and_return_unverified_module(data: &[u8]) -> Result<Self> {
        let mut ptr: c_lib::hlo_module_proto = std::ptr::null_mut();
        let status = unsafe {
            c_lib::hlo_module_proto_parse_and_return_unverified_module(
                data.as_ptr() as *const libc::c_char,
                data.len(),
                &mut ptr,
            )
        };
        handle_status(status)?;
        Ok(Self(ptr))
    }

    pub fn parse_proto(data: &[u8], binary: bool) -> Result<Self> {
        let mut ptr: c_lib::hlo_module_proto = std::ptr::null_mut();
        let status = unsafe {
            c_lib::hlo_module_proto_parse_proto(
                data.as_ptr() as *const libc::c_char,
                data.len(),
                binary,
                &mut ptr,
            )
        };
        handle_status(status)?;
        Ok(Self(ptr))
    }

    /// The HLO module using the human readable text format.
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> Result<String> {
        let mut ptr: *mut libc::c_char = std::ptr::null_mut();
        let status = unsafe { c_lib::hlo_module_proto_to_string(self.0, &mut ptr) };
        handle_status(status)?;
        Ok(unsafe { c_ptr_to_string(ptr) })
    }

    /// The HLO module serialized as a protobuf using the text format (pbtxt).
    pub fn to_pbtxt(&self) -> Result<String> {
        let mut ptr: *mut libc::c_char = std::ptr::null_mut();
        let status = unsafe { c_lib::hlo_module_proto_to_pbtxt(self.0, &mut ptr) };
        handle_status(status)?;
        Ok(unsafe { c_ptr_to_string(ptr) })
    }

    /// The HLO module serialized as a protobuf using the binary format.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut ptr: *mut libc::c_char = std::ptr::null_mut();
        let mut len = 0usize;
        let status = unsafe { c_lib::hlo_module_proto_serialize(self.0, &mut ptr, &mut len) };
        handle_status(status)?;
        let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() };
        unsafe { libc::free(ptr as *mut libc::c_void) };
        Ok(data)
    }

    /// Convert the HLO module to StableHLO, using the MLIR text format.
    pub fn to_stablehlo_string(&self) -> Result<String> {
        let mut ptr: *mut libc::c_char = std::ptr::null_mut();
        let status = unsafe { c_lib::hlo_module_proto_to_stablehlo_string(self.0, &mut ptr) };
        handle_status(status)?;
        Ok(unsafe { c_ptr_to_string(ptr) })
    }

    /// Write the HLO module to a file using the human readable text format.
    pub fn to_text_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        std::fs::write(path.as_ref(), self.to_string()?)?;
        Ok(())
    }

    /// Write the HLO module to a proto file, either in binary or pbtxt format.
    pub fn to_proto_file<P: AsRef<std::path::Path>>(&self, path: P, binary: bool) -> Result<()> {
        if binary {
            std::fs::write(path.as_ref(), self.to_bytes()?)?;
        } else {
            std::fs::write(path.as_ref(), self.to_pbtxt()?)?;
        }
        Ok(())
    }
}

impl Drop for HloModuleProto {
    fn drop(&mut self) {
        unsafe { c_lib::hlo_module_proto_free(self.0) }
    }
}
