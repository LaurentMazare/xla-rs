use super::{ElementType, FromPrimitive, NativeType, PrimitiveType, Shape};
use crate::{c_lib, Error, Result};

/// A literal represent a value, typically a multi-dimensional array, stored on the host device.
pub struct Literal(pub(super) c_lib::literal);

impl Clone for Literal {
    fn clone(&self) -> Self {
        let v = unsafe { c_lib::literal_clone(self.0) };
        Self(v)
    }
}

impl Literal {
    /// Create an unitialized literal based on some primitive type and some dimensions.
    pub fn create_from_shape(ty: PrimitiveType, dims: &[usize]) -> Self {
        let dims: Vec<_> = dims.iter().map(|x| *x as i64).collect();
        let v = unsafe { c_lib::literal_create_from_shape(ty as i32, dims.as_ptr(), dims.len()) };
        Self(v)
    }

    /// Create an unitialized literal based on some primitive type, some dimensions, and some data.
    /// The data is untyped, i.e. it is a sequence of bytes represented as a slice of `u8` even if
    /// the primitive type is not `U8`.
    pub fn create_from_shape_and_untyped_data(
        ty: PrimitiveType,
        dims: &[usize],
        untyped_data: &[u8],
    ) -> Result<Self> {
        let dims64: Vec<_> = dims.iter().map(|x| *x as i64).collect();
        let v = unsafe {
            c_lib::literal_create_from_shape_and_data(
                ty as i32,
                dims64.as_ptr(),
                dims64.len(),
                untyped_data.as_ptr() as *const libc::c_void,
                untyped_data.len(),
            )
        };
        if v.is_null() {
            return Err(Error::CannotCreateLiteralWithData {
                data_len_in_bytes: untyped_data.len(),
                ty,
                dims: dims.to_vec(),
            });
        }
        Ok(Self(v))
    }

    /// Get the first element from a literal. This returns an error if type `T` is not the
    /// primitive type that the literal uses.
    pub fn get_first_element<T: NativeType + ElementType>(&self) -> Result<T> {
        let ty = self.ty()?;
        if ty != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch { on_device: ty, on_host: T::PRIMITIVE_TYPE })?
        }
        if self.element_count() == 0 {
            Err(Error::EmptyLiteral)?
        }
        let v = unsafe { T::literal_get_first_element(self.0) };
        Ok(v)
    }

    /// The number of elements stored in the literal.
    pub fn element_count(&self) -> usize {
        unsafe { c_lib::literal_element_count(self.0) as usize }
    }

    /// The primitive type used by element stored in this literal.
    pub fn element_type(&self) -> Result<PrimitiveType> {
        let ty = unsafe { c_lib::literal_element_type(self.0) };
        match FromPrimitive::from_i32(ty) {
            None => Err(Error::UnexpectedElementType(ty)),
            Some(ty) => Ok(ty),
        }
    }
    /// The primitive type used by element stored in this literal, shortcut for `element_type`.
    pub fn ty(&self) -> Result<PrimitiveType> {
        self.element_type()
    }

    /// The literal size in bytes, this is the same as `element_count` multiplied by
    /// `element_size_in_bytes`.
    pub fn size_bytes(&self) -> usize {
        unsafe { c_lib::literal_size_bytes(self.0) as usize }
    }

    /// The [`Shape`] of the literal, this contains information about the dimensions of the
    /// underlying array, as well as the primitive type of the array's elements.
    pub fn shape(&self) -> Result<Shape> {
        let mut out: c_lib::shape = std::ptr::null_mut();
        unsafe { c_lib::literal_shape(self.0, &mut out) };
        let rank = unsafe { c_lib::shape_dimensions_size(out) };
        let dimensions: Vec<_> =
            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(out, i) }).collect();
        let ty = unsafe { c_lib::shape_element_type(out) };
        let tuple_shapes_size = unsafe { c_lib::shape_tuple_shapes_size(out) };
        unsafe { c_lib::shape_free(out) };
        match FromPrimitive::from_i32(ty) {
            None => Err(Error::UnexpectedElementType(ty)),
            Some(ty) => Ok(Shape { ty, dimensions, tuple_shapes_size }),
        }
    }

    /// Copy the literal data to a slice. This returns an error if the primitive type used by the
    /// literal is not `T` or if the number of elements in the slice and literal are different.
    pub fn copy_raw_to<T: ElementType>(&self, dst: &mut [T]) -> Result<()> {
        let ty = self.ty()?;
        let element_count = self.element_count();
        if ty != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch { on_device: ty, on_host: T::PRIMITIVE_TYPE })?
        }
        if dst.len() > element_count {
            Err(Error::BinaryBufferIsTooLarge { element_count, buffer_len: dst.len() })?
        }
        unsafe {
            c_lib::literal_copy_to(
                self.0,
                dst.as_mut_ptr() as *mut libc::c_void,
                element_count * T::ELEMENT_SIZE_IN_BYTES,
            )
        };
        Ok(())
    }

    /// Copy data from a slice to the literal. This returns an error if the primitive type used
    /// by the literal is not `T` or if number of elements in the slice and the literal are
    /// different.
    pub fn copy_raw_from<T: ElementType>(&mut self, src: &[T]) -> Result<()> {
        let ty = self.ty()?;
        let element_count = self.element_count();
        if ty != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch { on_device: ty, on_host: T::PRIMITIVE_TYPE })?
        }
        if src.len() > element_count {
            Err(Error::BinaryBufferIsTooLarge { element_count, buffer_len: src.len() })?
        }
        unsafe {
            c_lib::literal_copy_from(
                self.0,
                src.as_ptr() as *const libc::c_void,
                element_count * T::ELEMENT_SIZE_IN_BYTES,
            )
        };
        Ok(())
    }

    /// Copy the values stored in the literal in a newly created vector. The data is flattened out
    /// for literals with more than one dimension.
    pub fn to_vec<T: ElementType>(&self) -> Result<Vec<T>> {
        let element_count = self.element_count();
        // Maybe we should use an uninitialized vec instead?
        let mut data = vec![T::ZERO; element_count];
        self.copy_raw_to(&mut data)?;
        Ok(data)
    }

    /// Create a literal from a scalar value, the resulting literal has zero dimensions and stores
    /// a single element.
    pub fn scalar<T: NativeType>(t: T) -> Self {
        let ptr = unsafe { T::create_r0(t) };
        Literal(ptr)
    }

    /// Create a literal from a slice of data, the resulting literal has one dimension which size
    /// is the same as the slice passed as argument.
    pub fn vec<T: NativeType>(f: &[T]) -> Self {
        let ptr = unsafe { T::create_r1(f.as_ptr(), f.len()) };
        Literal(ptr)
    }

    /// Create a new literal containing the same data but using a different shape. This returns an
    /// error if the number of elements in the literal is different from the product of the target
    /// dimension sizes.
    pub fn reshape(&self, dims: &[i64]) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status =
            unsafe { c_lib::literal_reshape(self.0, dims.as_ptr(), dims.len(), &mut result) };
        super::handle_status(status)?;
        Ok(Literal(result))
    }

    /// Create a new literal containing the data from the original literal casted to a new
    /// primitive type. The dimensions of the resulting literal are the same as the dimensions of
    /// the original literal.
    pub fn convert(&self, ty: PrimitiveType) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status = unsafe { c_lib::literal_convert(self.0, ty as i32, &mut result) };
        super::handle_status(status)?;
        Ok(Literal(result))
    }

    /// When the input is a tuple, return a vector of its elements. This replaces the original
    /// value by an empty tuple, no copy is performed.
    pub fn decompose_tuple(&mut self) -> Result<Vec<Literal>> {
        match self.shape()?.tuple_size() {
            None => Ok(vec![]),
            Some(tuple_len) => {
                let mut outputs = vec![std::ptr::null_mut::<c_lib::_literal>(); tuple_len];
                unsafe { c_lib::literal_decompose_tuple(self.0, outputs.as_mut_ptr(), tuple_len) };
                Ok(outputs.into_iter().map(Literal).collect())
            }
        }
    }
}

impl<T: NativeType> From<T> for Literal {
    fn from(f: T) -> Self {
        Literal::scalar(f)
    }
}

impl<T: NativeType> From<&[T]> for Literal {
    fn from(f: &[T]) -> Self {
        Literal::vec(f)
    }
}

impl Drop for Literal {
    fn drop(&mut self) {
        unsafe { c_lib::literal_free(self.0) }
    }
}
