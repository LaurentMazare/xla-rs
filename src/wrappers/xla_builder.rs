use super::{
    handle_status, FromPrimitive, Literal, NativeType, PrimitiveType, Shape, XlaComputation, XlaOp,
};
use crate::{c_lib, Error, Result};
use std::rc::Rc;

/// A builder is used to keep track of a computation graph while it's being built.
pub(super) struct XlaBuilderInternal(c_lib::xla_builder);

#[derive(Clone)]
pub struct XlaBuilder(Rc<XlaBuilderInternal>);

impl XlaBuilder {
    /// Create a new builder with the associated name, the name is only used for debugging
    /// purposes.
    pub fn new(name: &str) -> XlaBuilder {
        let name = std::ffi::CString::new(name).unwrap();
        let xla_builder = unsafe { c_lib::xla_builder_create(name.as_ptr()) };
        XlaBuilder(Rc::new(XlaBuilderInternal(xla_builder)))
    }

    fn ptr(&self) -> c_lib::xla_builder {
        self.0 .0
    }

    /// Build a computation from the specified root node. This can only be called once.
    pub fn build(&self, op: &XlaOp) -> Result<XlaComputation> {
        let mut result: c_lib::xla_computation = std::ptr::null_mut();
        let status = unsafe { c_lib::build(self.ptr(), op.op, &mut result) };
        handle_status(status)?;
        Ok(XlaComputation(result))
    }

    /// This returns `Ok(())` if the graph creation has not generated any error so far. Otherwise
    /// the first error is returned.
    pub fn first_error(&self) -> Result<()> {
        let status = unsafe { c_lib::first_error(self.ptr()) };
        handle_status(status)?;
        Ok(())
    }

    /// This returns `Ok(())` if the graph creation has not generated any error so far. Otherwise
    /// the current status is returned.
    pub fn get_current_status(&self) -> Result<()> {
        let status = unsafe { c_lib::get_current_status(self.ptr()) };
        handle_status(status)?;
        Ok(())
    }

    /// Create a node with a constant value defined by the specified literal.
    pub fn constant_literal(&self, literal: &Literal) -> XlaOp {
        let op = unsafe { c_lib::constant_literal(self.ptr(), literal.0) };
        XlaOp { op, builder: self.clone() }
    }

    /// Create a node with a constant scalar value using the type of the element that is passed as
    /// argument.
    pub fn constant_r0<T: NativeType>(&self, f: T) -> XlaOp {
        let op = unsafe { T::constant_r0(self.ptr(), f) };
        XlaOp { op, builder: self.clone() }
    }

    /// A shorter notation for `constant_r0`.
    pub fn c0<T: NativeType>(&self, f: T) -> XlaOp {
        self.constant_r0(f)
    }

    /// Create an input node with the specified type and dimensions. A literal has to be passed for
    /// each of the parameter in the graph when calling the `execute` function, the parameter
    /// number are specified as incrementing values from 0 and represent the index of the
    /// associated literal in the slice passed to `execute`.
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

    /// A one dimension constant node based on some slice stored on the host.
    pub fn constant_r1<T: NativeType>(&self, f: &[T]) -> XlaOp {
        let op = unsafe { T::constant_r1(self.ptr(), f.as_ptr(), f.len()) };
        XlaOp { op, builder: self.clone() }
    }

    /// Shorthand function for `constant_r1`.
    pub fn c1<T: NativeType>(&self, f: &[T]) -> XlaOp {
        self.constant_r1(f)
    }

    /// A scalar node with the zero value for the associated type.
    pub fn zero(&self, element_type: super::PrimitiveType) -> XlaOp {
        let op = unsafe { c_lib::op_zero(self.ptr(), element_type as i32) };
        XlaOp { op, builder: self.clone() }
    }

    /// A scalar node with the one value for the associated type.
    pub fn one(&self, element_type: super::PrimitiveType) -> XlaOp {
        let op = unsafe { c_lib::op_one(self.ptr(), element_type as i32) };
        XlaOp { op, builder: self.clone() }
    }

    /// A scalar node with the minimum value for the associated type.
    pub fn min_value(&self, element_type: super::PrimitiveType) -> XlaOp {
        let op = unsafe { c_lib::op_min_value(self.ptr(), element_type as i32) };
        XlaOp { op, builder: self.clone() }
    }

    /// A scalar node with the maximum value for the associated type.
    pub fn max_value(&self, element_type: super::PrimitiveType) -> XlaOp {
        let op = unsafe { c_lib::op_max_value(self.ptr(), element_type as i32) };
        XlaOp { op, builder: self.clone() }
    }

    /// A constant node with the specified shape that holds increasing values starting from 0 along
    /// the iota dimension.
    pub fn iota(
        &self,
        element_type: super::PrimitiveType,
        dims: &[i64],
        iota_dimension: i64,
    ) -> XlaOp {
        let op = unsafe {
            c_lib::op_iota(
                self.ptr(),
                element_type as i32,
                dims.len(),
                dims.as_ptr(),
                iota_dimension,
            )
        };
        XlaOp { op, builder: self.clone() }
    }

    /// A constant node for a unidimensional array of increasing values starting from 0.
    pub fn iota1(&self, element_type: super::PrimitiveType, size: usize) -> XlaOp {
        let op = unsafe { c_lib::op_iota1(self.ptr(), element_type as i32, size) };
        XlaOp { op, builder: self.clone() }
    }

    /// An error node, using the 'internal error' error type.
    pub fn internal_error(&self, msg: &str) -> XlaOp {
        let msg = std::ffi::CString::new(msg).unwrap();
        let op = unsafe { c_lib::op_internal_error(self.ptr(), msg.as_ptr()) };
        XlaOp { op, builder: self.clone() }
    }

    /// An error node, using the 'unknown error' error type.
    pub fn unknown_error(&self, msg: &str) -> XlaOp {
        let msg = std::ffi::CString::new(msg).unwrap();
        let op = unsafe { c_lib::op_unknown_error(self.ptr(), msg.as_ptr()) };
        XlaOp { op, builder: self.clone() }
    }

    /// An error node, using the 'invalid argument error' error type.
    pub fn invalid_argument_error(&self, msg: &str) -> XlaOp {
        let msg = std::ffi::CString::new(msg).unwrap();
        let op = unsafe { c_lib::op_invalid_argument_error(self.ptr(), msg.as_ptr()) };
        XlaOp { op, builder: self.clone() }
    }

    /// Wrap a potential error in an error node. If the argument is `Ok(op)` then `op` is passed
    /// back as the result.
    pub fn wrap_error(&self, op: Result<XlaOp>) -> XlaOp {
        match op {
            Ok(op) => op,
            Err(err) => self.internal_error(&err.to_string()),
        }
    }

    /// The shape associated with this op.
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

    /// The dimension sizes associated with this op.
    pub fn get_dims(&self, op: &XlaOp) -> Result<Vec<usize>> {
        let rank = self.get_dimensions_size(op)?;
        let mut dims = vec![0; rank];
        let status = unsafe { c_lib::get_dimensions(self.ptr(), op.op, dims.as_mut_ptr()) };
        handle_status(status)?;
        Ok(dims)
    }

    /// The element type associated with this op.
    pub fn get_element_type(&self, op: &XlaOp) -> Result<super::PrimitiveType> {
        let mut element_type = 0i32;
        let status = unsafe { c_lib::get_element_type(self.ptr(), op.op, &mut element_type) };
        handle_status(status)?;
        FromPrimitive::from_i32(element_type).ok_or(Error::UnexpectedElementType(element_type))
    }

    /// The number of dimensions (a.k.a the rank) associated with this op.
    pub fn get_dimensions_size(&self, op: &XlaOp) -> Result<usize> {
        let mut dsize = 0i32;
        let status = unsafe { c_lib::get_dimensions_size(self.ptr(), op.op, &mut dsize) };
        handle_status(status)?;
        Ok(dsize as usize)
    }
}

impl Drop for XlaBuilderInternal {
    fn drop(&mut self) {
        unsafe { c_lib::xla_builder_free(self.0) }
    }
}
