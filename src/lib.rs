use anyhow::{anyhow, Result};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use std::marker::PhantomData;
mod c_lib;

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

impl Shape {
    pub fn new(element_type: PrimitiveType, dimensions: Vec<i64>) -> Shape {
        Shape {
            element_type,
            dimensions,
        }
    }
}

pub struct XlaBuilder(c_lib::xla_builder);
pub struct XlaOp<'a> {
    op: c_lib::xla_op,
    marker: PhantomData<&'a XlaBuilder>,
}
pub struct Literal(c_lib::literal);
pub struct LiteralSlice(c_lib::literal_slice);
pub struct GlobalData(c_lib::global_data);

fn handle_status(status: c_lib::status) -> Result<()> {
    if status.is_null() {
        Ok(())
    } else {
        let error_message_ptr = unsafe { c_lib::status_error_message(status) };
        let error_message = unsafe { std::ffi::CStr::from_ptr(error_message_ptr) }
            .to_string_lossy()
            .into_owned();
        unsafe { libc::free(error_message_ptr as *mut libc::c_void) };
        unsafe { c_lib::status_free(status) };
        Err(anyhow!(error_message))
    }
}

impl XlaBuilder {
    pub fn new(name: &str) -> XlaBuilder {
        let xla_builder = unsafe { c_lib::xla_builder_create(name.as_ptr() as *const i8) };
        XlaBuilder(xla_builder)
    }

    pub fn run(&self, op: &XlaOp, args: &[GlobalData]) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.0).collect();
        let status =
            unsafe { c_lib::run(self.0, op.op, args.as_ptr(), args.len() as i32, &mut result) };
        handle_status(status)?;
        Ok(Literal(result))
    }

    pub fn constant_r0(&self, f: f32) -> XlaOp {
        let op = unsafe { c_lib::constant_r0_float(self.0, f) };
        XlaOp {
            op,
            marker: PhantomData,
        }
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
        XlaOp {
            op,
            marker: PhantomData,
        }
    }

    pub fn constant_r1(&self, len: i64, f: f32) -> XlaOp {
        let op = unsafe { c_lib::constant_r1_float(self.0, len, f) };
        XlaOp {
            op,
            marker: PhantomData,
        }
    }

    pub fn get_shape(&self, op: &XlaOp) -> Result<Shape> {
        let mut out: c_lib::shape = std::ptr::null_mut();
        let status = unsafe { c_lib::get_shape(self.0, op.op, &mut out) };
        handle_status(status)?;
        let rank = unsafe { c_lib::shape_dimensions_size(out) };
        let dimensions: Vec<_> = (0..rank)
            .map(|i| unsafe { c_lib::shape_dimensions(out, i) })
            .collect();
        let element_type = FromPrimitive::from_i32(unsafe { c_lib::shape_element_type(out) });
        unsafe { c_lib::shape_free(out) };
        match element_type {
            None => Err(anyhow!("unexpected element type")),
            Some(element_type) => Ok(Shape {
                element_type,
                dimensions,
            }),
        }
    }
}

impl XlaOp<'_> {
    pub fn add(&self, op: &XlaOp) -> XlaOp {
        let op = unsafe { c_lib::add(self.op, op.op) };
        XlaOp {
            op,
            marker: PhantomData,
        }
    }
}

impl Literal {
    pub fn get_first_element_f32(&self) -> f32 {
        unsafe { c_lib::literal_get_first_element_f32(self.0) }
    }
}

impl LiteralSlice {
    pub fn transfer_to_server(&self) -> Result<GlobalData> {
        let mut result: c_lib::global_data = std::ptr::null_mut();
        let status = unsafe { c_lib::transfer_to_server(self.0, &mut result) };
        handle_status(status)?;
        Ok(GlobalData(result))
    }
}

impl GlobalData {
    pub fn transfer(&self) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status = unsafe { c_lib::transfer(self.0, &mut result) };
        handle_status(status)?;
        Ok(Literal(result))
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

impl Drop for LiteralSlice {
    fn drop(&mut self) {
        unsafe { c_lib::literal_slice_free(self.0) }
    }
}

impl Drop for GlobalData {
    fn drop(&mut self) {
        unsafe { c_lib::global_data_free(self.0) }
    }
}
