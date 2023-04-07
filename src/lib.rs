use anyhow::{anyhow, Result};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use std::marker::PhantomData;
mod c_lib;

unsafe fn c_ptr_to_string(ptr: *const std::ffi::c_char) -> String {
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

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape {
    element_type: PrimitiveType,
    dimensions: Vec<i64>,
}

pub trait ElementType {
    const PRIMITIVE_TYPE: PrimitiveType;
}

impl ElementType for f32 {
    const PRIMITIVE_TYPE: PrimitiveType = PrimitiveType::F32;
}

impl Shape {
    pub fn new<E: ElementType>(dimensions: Vec<i64>) -> Shape {
        Shape { element_type: E::PRIMITIVE_TYPE, dimensions }
    }
}

pub struct XlaBuilder(c_lib::xla_builder);
pub struct XlaComputation(c_lib::xla_computation);
pub struct PjRtClient(c_lib::pjrt_client);
pub struct PjRtBuffer(c_lib::pjrt_buffer);
pub struct PjRtLoadedExecutable(c_lib::pjrt_loaded_executable);
pub struct PjRtDevice<'a> {
    device: c_lib::pjrt_device,
    marker: PhantomData<&'a PjRtClient>,
}

pub struct XlaOp<'a> {
    op: c_lib::xla_op,
    marker: PhantomData<&'a XlaBuilder>,
}
pub struct Literal(c_lib::literal);

impl AsRef<Literal> for Literal {
    fn as_ref(&self) -> &Literal {
        self
    }
}

impl AsRef<PjRtBuffer> for PjRtBuffer {
    fn as_ref(&self) -> &PjRtBuffer {
        self
    }
}

fn handle_status(status: c_lib::status) -> Result<()> {
    if status.is_null() {
        Ok(())
    } else {
        let error_message = unsafe {
            let error_message_ptr = c_lib::status_error_message(status);
            let error_message = c_ptr_to_string(error_message_ptr);
            c_lib::status_free(status);
            error_message
        };
        Err(anyhow!(error_message))
    }
}

impl PjRtClient {
    pub fn cpu() -> Result<Self> {
        let mut result: c_lib::pjrt_client = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_client_create(&mut result) };
        handle_status(status)?;
        Ok(Self(result))
    }

    pub fn compile(&self, c: &XlaComputation) -> Result<PjRtLoadedExecutable> {
        let mut result: c_lib::pjrt_loaded_executable = std::ptr::null_mut();
        let status = unsafe { c_lib::compile(self.0, c.0, &mut result) };
        handle_status(status)?;
        Ok(PjRtLoadedExecutable(result))
    }

    pub fn device_count(&self) -> usize {
        unsafe { c_lib::pjrt_client_device_count(self.0) as usize }
    }

    pub fn addressable_device_count(&self) -> usize {
        unsafe { c_lib::pjrt_client_addressable_device_count(self.0) as usize }
    }

    pub fn platform_name(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_client_platform_name(self.0);
            c_ptr_to_string(ptr)
        }
    }

    pub fn platform_version(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_client_platform_version(self.0);
            c_ptr_to_string(ptr)
        }
    }

    pub fn devices(&self) -> Vec<PjRtDevice> {
        let device_count = self.device_count();
        let mut device_ptrs = vec![std::ptr::null_mut(); device_count];
        unsafe { c_lib::pjrt_client_devices(self.0, device_ptrs.as_mut_ptr()) };
        device_ptrs.into_iter().map(|device| PjRtDevice { device, marker: PhantomData }).collect()
    }

    pub fn addressable_devices(&self) -> Vec<PjRtDevice> {
        let device_count = self.addressable_device_count();
        let mut device_ptrs = vec![std::ptr::null_mut(); device_count];
        unsafe { c_lib::pjrt_client_addressable_devices(self.0, device_ptrs.as_mut_ptr()) };
        device_ptrs.into_iter().map(|device| PjRtDevice { device, marker: PhantomData }).collect()
    }
}

impl<'a> PjRtDevice<'a> {
    pub fn id(&self) -> usize {
        (unsafe { c_lib::pjrt_device_id(self.device) }) as usize
    }

    pub fn process_index(&self) -> usize {
        (unsafe { c_lib::pjrt_device_process_index(self.device) }) as usize
    }

    pub fn local_hardware_id(&self) -> usize {
        (unsafe { c_lib::pjrt_device_local_hardware_id(self.device) }) as usize
    }

    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_device_to_string(self.device);
            c_ptr_to_string(ptr)
        }
    }

    pub fn kind(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_device_kind(self.device);
            c_ptr_to_string(ptr)
        }
    }

    pub fn debug_string(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_device_debug_string(self.device);
            c_ptr_to_string(ptr)
        }
    }
}

impl PjRtBuffer {
    pub fn copy_to_device(&self, device: PjRtDevice<'_>) -> Result<Self> {
        let mut result: c_lib::pjrt_buffer = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_buffer_copy_to_device(self.0, device.device, &mut result) };
        handle_status(status)?;
        Ok(Self(result))
    }

    pub fn to_literal_sync(&self) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_buffer_to_literal_sync(self.0, &mut result) };
        handle_status(status)?;
        Ok(Literal(result))
    }
}

impl PjRtLoadedExecutable {
    fn process_execute_outputs(outputs: *mut *mut c_lib::pjrt_buffer) -> Vec<Vec<PjRtBuffer>> {
        unsafe {
            let mut vec = vec![];
            loop {
                let outputs = *outputs.add(vec.len());
                if outputs.is_null() {
                    break;
                }
                let mut replica_vec = vec![];
                loop {
                    let outputs = *outputs.add(replica_vec.len());
                    if outputs.is_null() {
                        break;
                    }
                    replica_vec.push(PjRtBuffer(outputs));
                }
                libc::free(outputs as *mut libc::c_void);
                vec.push(replica_vec);
            }
            libc::free(outputs as *mut libc::c_void);
            vec
        }
    }

    pub fn execute<P: AsRef<PjRtBuffer>>(&self, args: &[P]) -> Result<Vec<Vec<PjRtBuffer>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.as_ref().0).collect();
        let status =
            unsafe { c_lib::execute(self.0, args.as_ptr(), args.len() as i32, &mut outputs) };
        handle_status(status)?;
        Ok(Self::process_execute_outputs(outputs))
    }

    pub fn execute_literal<L: AsRef<Literal>>(&self, args: &[L]) -> Result<Vec<Vec<PjRtBuffer>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.as_ref().0).collect();
        let status = unsafe {
            c_lib::execute_literal(self.0, args.as_ptr(), args.len() as i32, &mut outputs)
        };
        handle_status(status)?;
        Ok(Self::process_execute_outputs(outputs))
    }
}

impl XlaBuilder {
    pub fn new(name: &str) -> XlaBuilder {
        let name = std::ffi::CString::new(name).unwrap();
        let xla_builder = unsafe { c_lib::xla_builder_create(name.as_ptr()) };
        XlaBuilder(xla_builder)
    }

    pub fn build(&self, op: &XlaOp) -> Result<XlaComputation> {
        let mut result: c_lib::xla_computation = std::ptr::null_mut();
        let status = unsafe { c_lib::build(self.0, op.op, &mut result) };
        handle_status(status)?;
        Ok(XlaComputation(result))
    }

    pub fn constant_r0(&self, f: f32) -> XlaOp {
        let op = unsafe { c_lib::constant_r0_float(self.0, f) };
        XlaOp { op, marker: PhantomData }
    }

    pub fn parameter(&self, id: i64, shape: &Shape, name: &str) -> XlaOp {
        let op = unsafe {
            c_lib::parameter(
                self.0,
                id,
                shape.element_type as i32,
                shape.dimensions.len() as i32,
                shape.dimensions.as_ptr(),
                name.as_ptr() as *const libc::c_char,
            )
        };
        XlaOp { op, marker: PhantomData }
    }

    pub fn constant_r1(&self, len: i64, f: f32) -> XlaOp {
        let op = unsafe { c_lib::constant_r1_float(self.0, len, f) };
        XlaOp { op, marker: PhantomData }
    }

    pub fn get_shape(&self, op: &XlaOp) -> Result<Shape> {
        let mut out: c_lib::shape = std::ptr::null_mut();
        let status = unsafe { c_lib::get_shape(self.0, op.op, &mut out) };
        handle_status(status)?;
        let rank = unsafe { c_lib::shape_dimensions_size(out) };
        let dimensions: Vec<_> =
            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(out, i) }).collect();
        let element_type = FromPrimitive::from_i32(unsafe { c_lib::shape_element_type(out) });
        unsafe { c_lib::shape_free(out) };
        match element_type {
            None => Err(anyhow!("unexpected element type")),
            Some(element_type) => Ok(Shape { element_type, dimensions }),
        }
    }
}

macro_rules! binary_op {
    ($func_name:ident, $expression:expr) => {
        pub fn $func_name(&self, op: &XlaOp) -> XlaOp {
            let op = unsafe { $expression(self.op, op.op) };
            XlaOp { op, marker: PhantomData }
        }
    };
}

macro_rules! unary_op {
    ($func_name:ident, $expression:expr) => {
        pub fn $func_name(&self) -> XlaOp {
            let op = unsafe { $expression(self.op) };
            XlaOp { op, marker: PhantomData }
        }
    };
}

impl XlaOp<'_> {
    binary_op!(add, c_lib::op_add);
    binary_op!(sub, c_lib::op_sub);
    binary_op!(mul, c_lib::op_mul);
    binary_op!(div, c_lib::op_div);
    binary_op!(rem, c_lib::op_rem);
    binary_op!(max, c_lib::op_max);
    binary_op!(min, c_lib::op_min);
    binary_op!(and, c_lib::op_and);
    binary_op!(or, c_lib::op_or);
    binary_op!(xor, c_lib::op_xor);
    binary_op!(atan2, c_lib::op_atan2);
    binary_op!(pow, c_lib::op_pow);
    binary_op!(dot, c_lib::op_dot);

    unary_op!(not, c_lib::op_not);
    unary_op!(abs, c_lib::op_abs);
    unary_op!(exp, c_lib::op_exp);
    unary_op!(expm1, c_lib::op_expm1);
    unary_op!(floor, c_lib::op_floor);
    unary_op!(ceil, c_lib::op_ceil);
    unary_op!(round, c_lib::op_round);
    unary_op!(log, c_lib::op_log);
    unary_op!(log1p, c_lib::op_log1p);
    unary_op!(logistic, c_lib::op_logistic);
    unary_op!(sign, c_lib::op_sign);
    unary_op!(clz, c_lib::op_clz);
    unary_op!(cos, c_lib::op_cos);
    unary_op!(sin, c_lib::op_sin);
    unary_op!(tanh, c_lib::op_tanh);
    unary_op!(real, c_lib::op_real);
    unary_op!(imag, c_lib::op_imag);
    unary_op!(sqrt, c_lib::op_sqrt);
    unary_op!(rsqrt, c_lib::op_rsqrt);
    unary_op!(cbrt, c_lib::op_cbrt);
    unary_op!(is_finite, c_lib::op_is_finite);
    unary_op!(neg, c_lib::op_neg);
}

impl Literal {
    pub fn get_first_element_f32(&self) -> f32 {
        unsafe { c_lib::literal_get_first_element_f32(self.0) }
    }

    pub fn element_count(&self) -> usize {
        unsafe { c_lib::literal_element_count(self.0) as usize }
    }

    pub fn shape(&self) -> Result<Shape> {
        let mut out: c_lib::shape = std::ptr::null_mut();
        unsafe { c_lib::literal_shape(self.0, &mut out) };
        let rank = unsafe { c_lib::shape_dimensions_size(out) };
        let dimensions: Vec<_> =
            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(out, i) }).collect();
        let element_type = FromPrimitive::from_i32(unsafe { c_lib::shape_element_type(out) });
        unsafe { c_lib::shape_free(out) };
        match element_type {
            None => Err(anyhow!("unexpected element type")),
            Some(element_type) => Ok(Shape { element_type, dimensions }),
        }
    }
}

impl From<f32> for Literal {
    fn from(f: f32) -> Self {
        let ptr = unsafe { c_lib::create_r0_f32(f) };
        Literal(ptr)
    }
}

impl From<&[f32]> for Literal {
    fn from(f: &[f32]) -> Self {
        let ptr = unsafe { c_lib::create_r1_f32(f.as_ptr(), f.len() as i32) };
        Literal(ptr)
    }
}

impl Drop for XlaBuilder {
    fn drop(&mut self) {
        unsafe { c_lib::xla_builder_free(self.0) }
    }
}

impl Drop for XlaOp<'_> {
    fn drop(&mut self) {
        unsafe { c_lib::xla_op_free(self.op) }
    }
}

impl Drop for Literal {
    fn drop(&mut self) {
        unsafe { c_lib::literal_free(self.0) }
    }
}

impl Drop for XlaComputation {
    fn drop(&mut self) {
        unsafe { c_lib::xla_computation_free(self.0) }
    }
}

impl Drop for PjRtClient {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_client_free(self.0) }
    }
}

impl Drop for PjRtLoadedExecutable {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_loaded_executable_free(self.0) }
    }
}

impl Drop for PjRtBuffer {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_buffer_free(self.0) }
    }
}
