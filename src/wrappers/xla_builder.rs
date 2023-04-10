use super::{
    handle_status, FromPrimitive, Literal, NativeType, PrimitiveType, Shape, XlaComputation, XlaOp,
};
use crate::{c_lib, Error, Result};
use std::rc::Rc;

pub(super) struct XlaBuilderInternal(c_lib::xla_builder);

#[derive(Clone)]
pub struct XlaBuilder(Rc<XlaBuilderInternal>);

impl XlaBuilder {
    pub fn new(name: &str) -> XlaBuilder {
        let name = std::ffi::CString::new(name).unwrap();
        let xla_builder = unsafe { c_lib::xla_builder_create(name.as_ptr()) };
        XlaBuilder(Rc::new(XlaBuilderInternal(xla_builder)))
    }

    fn ptr(&self) -> c_lib::xla_builder {
        self.0 .0
    }

    pub fn build(&self, op: &XlaOp) -> Result<XlaComputation> {
        let mut result: c_lib::xla_computation = std::ptr::null_mut();
        let status = unsafe { c_lib::build(self.ptr(), op.op, &mut result) };
        handle_status(status)?;
        Ok(XlaComputation(result))
    }

    pub fn constant_literal(&self, literal: Literal) -> XlaOp {
        let op = unsafe { c_lib::constant_literal(self.ptr(), literal.0) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn constant_r0<T: NativeType>(&self, f: T) -> XlaOp {
        let op = unsafe { T::constant_r0(self.ptr(), f) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn c0<T: NativeType>(&self, f: T) -> XlaOp {
        self.constant_r0(f)
    }

    pub fn parameter(
        &self,
        parameter_number: i64,
        element_type: PrimitiveType,
        dims: &[i64],
        name: &str,
    ) -> XlaOp {
        let op = unsafe {
            c_lib::parameter(
                self.ptr(),
                parameter_number,
                element_type as i32,
                dims.len() as i32,
                dims.as_ptr(),
                name.as_ptr() as *const libc::c_char,
            )
        };
        XlaOp { op, builder: self.clone() }
    }

    pub fn parameter_with_shape(&self, parameter_number: i64, shape: &Shape, name: &str) -> XlaOp {
        self.parameter(parameter_number, shape.element_type, &shape.dimensions, name)
    }

    pub fn constant_r1c<T: NativeType>(&self, f: T, len: usize) -> XlaOp {
        let op = unsafe { T::constant_r1c(self.ptr(), f, len) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn constant_r1<T: NativeType>(&self, f: &[T]) -> XlaOp {
        let op = unsafe { T::constant_r1(self.ptr(), f.as_ptr(), f.len()) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn c1<T: NativeType>(&self, f: &[T]) -> XlaOp {
        self.constant_r1(f)
    }

    pub fn zero(&self, element_type: super::PrimitiveType) -> XlaOp {
        let op = unsafe { c_lib::op_zero(self.ptr(), element_type as i32) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn min_value(&self, element_type: super::PrimitiveType) -> XlaOp {
        let op = unsafe { c_lib::op_min_value(self.ptr(), element_type as i32) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn max_value(&self, element_type: super::PrimitiveType) -> XlaOp {
        let op = unsafe { c_lib::op_max_value(self.ptr(), element_type as i32) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn internal_error(&self, msg: &str) -> XlaOp {
        let msg = std::ffi::CString::new(msg).unwrap();
        let op = unsafe { c_lib::op_internal_error(self.ptr(), msg.as_ptr()) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn unknown_error(&self, msg: &str) -> XlaOp {
        let msg = std::ffi::CString::new(msg).unwrap();
        let op = unsafe { c_lib::op_unknown_error(self.ptr(), msg.as_ptr()) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn invalid_argument_error(&self, msg: &str) -> XlaOp {
        let msg = std::ffi::CString::new(msg).unwrap();
        let op = unsafe { c_lib::op_invalid_argument_error(self.ptr(), msg.as_ptr()) };
        XlaOp { op, builder: self.clone() }
    }

    pub fn wrap_error(&self, op: Result<XlaOp>) -> XlaOp {
        match op {
            Ok(op) => op,
            Err(err) => self.internal_error(&err.to_string()),
        }
    }

    pub fn get_shape(&self, op: &XlaOp) -> Result<Shape> {
        let mut out: c_lib::shape = std::ptr::null_mut();
        let status = unsafe { c_lib::get_shape(self.ptr(), op.op, &mut out) };
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

    pub fn get_element_type(&self, op: &XlaOp) -> Result<super::PrimitiveType> {
        let mut element_type = 0i32;
        let status = unsafe { c_lib::get_element_type(self.ptr(), op.op, &mut element_type) };
        handle_status(status)?;
        FromPrimitive::from_i32(element_type).ok_or(Error::UnexpectedElementType(element_type))
    }
}

impl Drop for XlaBuilderInternal {
    fn drop(&mut self) {
        unsafe { c_lib::xla_builder_free(self.0) }
    }
}
