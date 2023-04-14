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
pub use pjrt_client::PjRtClient;
pub use pjrt_device::PjRtDevice;
pub use pjrt_loaded_executable::PjRtLoadedExecutable;
pub use shape::Shape;
pub use xla_builder::XlaBuilder;
pub use xla_op::XlaOp;

pub(self) unsafe fn c_ptr_to_string(ptr: *const std::ffi::c_char) -> String {
    let str = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
    libc::free(ptr as *mut libc::c_void);
    str
}

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
    C64 = 15,
    C128 = 18,
    Tuple = 13,
    OpaqueType = 14,
    Token = 17,
}

impl PrimitiveType {
    pub fn element_size_in_bytes(&self) -> Option<usize> {
        match self {
            PrimitiveType::Invalid => None,
            PrimitiveType::Pred => None,
            PrimitiveType::S8 => Some(1),
            PrimitiveType::S16 => Some(2),
            PrimitiveType::S32 => Some(4),
            PrimitiveType::S64 => Some(8),
            PrimitiveType::U8 => Some(1),
            PrimitiveType::U16 => Some(2),
            PrimitiveType::U32 => Some(4),
            PrimitiveType::U64 => Some(8),
            PrimitiveType::F16 => Some(2),
            PrimitiveType::F32 => Some(4),
            PrimitiveType::Bf16 => Some(2),
            PrimitiveType::F64 => Some(8),
            PrimitiveType::C64 => Some(8),
            PrimitiveType::C128 => Some(16),
            PrimitiveType::Tuple => None,
            PrimitiveType::OpaqueType => None,
            PrimitiveType::Token => None,
        }
    }
}

pub trait ElementType: Copy {
    const PRIMITIVE_TYPE: PrimitiveType;
    const ELEMENT_SIZE_IN_BYTES: usize;
    const ZERO: Self;
}

#[allow(clippy::missing_safety_doc)]
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
        impl ElementType for $ty {
            const PRIMITIVE_TYPE: PrimitiveType = PrimitiveType::$v;
            const ELEMENT_SIZE_IN_BYTES: usize = $sz;
            const ZERO: Self = 0 as Self;
        }
    };
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

pub struct XlaComputation(c_lib::xla_computation);

pub(self) fn handle_status(status: c_lib::status) -> Result<()> {
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
    pub fn name(&self) -> String {
        unsafe {
            let ptr = c_lib::xla_computation_name(self.0);
            c_ptr_to_string(ptr)
        }
    }

    pub fn compile(&self, client: &PjRtClient) -> Result<PjRtLoadedExecutable> {
        client.compile(self)
    }
}

impl Drop for XlaComputation {
    fn drop(&mut self) {
        unsafe { c_lib::xla_computation_free(self.0) }
    }
}
