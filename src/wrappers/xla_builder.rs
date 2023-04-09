use super::{handle_status, FromPrimitive, Literal, NativeType, Shape, XlaComputation, XlaOp};
use crate::{c_lib, Error, Result};
use std::marker::PhantomData;

pub struct XlaBuilder(c_lib::xla_builder);

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

    pub fn constant_literal(&self, literal: Literal) -> XlaOp {
        let op = unsafe { c_lib::constant_literal(self.0, literal.0) };
        XlaOp { op, marker: PhantomData }
    }

    pub fn constant_r0<T: NativeType>(&self, f: T) -> XlaOp {
        let op = unsafe { T::constant_r0(self.0, f) };
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

    pub fn constant_r1c<T: NativeType>(&self, f: T, len: usize) -> XlaOp {
        let op = unsafe { T::constant_r1c(self.0, f, len) };
        XlaOp { op, marker: PhantomData }
    }

    pub fn constant_r1<T: NativeType>(&self, f: &[T]) -> XlaOp {
        let op = unsafe { T::constant_r1(self.0, f.as_ptr(), f.len()) };
        XlaOp { op, marker: PhantomData }
    }

    pub fn get_shape(&self, op: &XlaOp) -> Result<Shape> {
        let mut out: c_lib::shape = std::ptr::null_mut();
        let status = unsafe { c_lib::get_shape(self.0, op.op, &mut out) };
        handle_status(status)?;
        let rank = unsafe { c_lib::shape_dimensions_size(out) };
        let dimensions: Vec<_> =
            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(out, i) }).collect();
        let element_type = unsafe { c_lib::shape_element_type(out) };
        unsafe { c_lib::shape_free(out) };
        match FromPrimitive::from_i32(element_type) {
            None => Err(Error::UnexpectedElementType(element_type)),
            Some(element_type) => Ok(Shape { element_type, dimensions }),
        }
    }
}

impl Drop for XlaBuilder {
    fn drop(&mut self) {
        unsafe { c_lib::xla_builder_free(self.0) }
    }
}
