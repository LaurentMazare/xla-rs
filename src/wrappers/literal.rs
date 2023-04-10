use super::{ElementType, FromPrimitive, NativeType, PrimitiveType, Shape};
use crate::{c_lib, Error, Result};

pub struct Literal(pub(super) c_lib::literal);

impl Literal {
    pub fn get_first_element<T: NativeType + ElementType>(&self) -> Result<T> {
        let element_type = self.element_type()?;
        if element_type != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch { on_device: element_type, on_host: T::PRIMITIVE_TYPE })?
        }
        if self.element_count() == 0 {
            Err(Error::EmptyLiteral)?
        }
        let v = unsafe { T::literal_get_first_element(self.0) };
        Ok(v)
    }

    pub fn element_count(&self) -> usize {
        unsafe { c_lib::literal_element_count(self.0) as usize }
    }

    pub fn element_type(&self) -> Result<PrimitiveType> {
        let element_type = unsafe { c_lib::literal_element_type(self.0) };
        match FromPrimitive::from_i32(element_type) {
            None => Err(Error::UnexpectedElementType(element_type)),
            Some(element_type) => Ok(element_type),
        }
    }

    pub fn size_bytes(&self) -> usize {
        unsafe { c_lib::literal_size_bytes(self.0) as usize }
    }

    pub fn shape(&self) -> Result<Shape> {
        let mut out: c_lib::shape = std::ptr::null_mut();
        unsafe { c_lib::literal_shape(self.0, &mut out) };
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

    pub fn copy_raw<T: ElementType>(&self, dst: &mut [T]) -> Result<()> {
        let element_type = self.element_type()?;
        let element_count = self.element_count();
        if element_type != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch { on_device: element_type, on_host: T::PRIMITIVE_TYPE })?
        }
        if dst.len() > element_count {
            Err(Error::BinaryBufferIsTooLarge { element_count, buffer_len: dst.len() })?
        }
        unsafe {
            c_lib::literal_copy(
                self.0,
                dst.as_mut_ptr() as *mut libc::c_void,
                element_count * T::ELEMENT_SIZE_IN_BYTES,
            )
        };
        Ok(())
    }

    pub fn to_vec<T: ElementType>(&self) -> Result<Vec<T>> {
        let element_count = self.element_count();
        // Maybe we should use an uninitialized vec instead?
        let mut data = vec![T::ZERO; element_count];
        self.copy_raw(&mut data)?;
        Ok(data)
    }

    pub fn scalar<T: NativeType>(t: T) -> Self {
        let ptr = unsafe { T::create_r0(t) };
        Literal(ptr)
    }

    pub fn vec<T: NativeType>(f: &[T]) -> Self {
        let ptr = unsafe { T::create_r1(f.as_ptr(), f.len()) };
        Literal(ptr)
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