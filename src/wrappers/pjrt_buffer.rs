use super::{ElementType, FromPrimitive, Literal, PjRtDevice, Shape};
use crate::{c_lib, Error, Result};

pub struct PjRtBuffer(pub(super) c_lib::pjrt_buffer);

impl PjRtBuffer {
    pub fn copy_to_device(&self, device: PjRtDevice<'_>) -> Result<Self> {
        let mut result: c_lib::pjrt_buffer = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_buffer_copy_to_device(self.0, device.device, &mut result) };
        super::handle_status(status)?;
        Ok(Self(result))
    }

    pub fn to_literal_sync(&self) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_buffer_to_literal_sync(self.0, &mut result) };
        super::handle_status(status)?;
        Ok(Literal(result))
    }

    pub fn on_device_shape(&self) -> Result<Shape> {
        let shape = unsafe { c_lib::pjrt_buffer_on_device_shape(self.0) };
        let rank = unsafe { c_lib::shape_dimensions_size(shape) };
        let dimensions: Vec<_> =
            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(shape, i) }).collect();
        let element_type = unsafe { c_lib::shape_element_type(shape) };
        unsafe { c_lib::shape_free(shape) };
        match FromPrimitive::from_i32(element_type) {
            None => Err(Error::UnexpectedElementType(element_type)),
            Some(element_type) => Ok(Shape { element_type, dimensions }),
        }
    }

    pub fn copy_raw_to_host_sync<T: ElementType>(
        &self,
        dst: &mut [T],
        offset: usize,
    ) -> Result<()> {
        let shape = self.on_device_shape()?;
        if shape.element_type != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch {
                on_device: shape.element_type,
                on_host: T::PRIMITIVE_TYPE,
            })?
        }
        if offset + dst.len() > shape.size() {
            Err(Error::TargetBufferIsTooLarge { offset, shape, buffer_len: dst.len() })?
        }
        let status = unsafe {
            c_lib::pjrt_buffer_copy_raw_to_host_sync(
                self.0,
                dst.as_mut_ptr() as *mut libc::c_void,
                offset,
                dst.len() * T::ELEMENT_SIZE_IN_BYTES,
            )
        };
        super::handle_status(status)?;
        Ok(())
    }
}

impl Drop for PjRtBuffer {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_buffer_free(self.0) }
    }
}
