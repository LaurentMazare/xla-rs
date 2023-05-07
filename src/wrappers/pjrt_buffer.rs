//! A view on a memory slice hosted on a device.
use super::{ArrayElement, FromPrimitive, Literal, PjRtDevice, Shape};
use crate::{c_lib, Error, Result};

/// A buffer represents a view on a memory slice hosted on a device.
pub struct PjRtBuffer {
    pub(super) buffer: c_lib::pjrt_buffer,
    pub(super) client: super::PjRtClient,
}

impl PjRtBuffer {
    /// The client that owns this buffer.
    pub fn client(&self) -> &super::PjRtClient {
        &self.client
    }

    /// Copy the buffer to a different device.
    pub fn copy_to_device(&self, device: PjRtDevice) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_buffer_copy_to_device(self.buffer, device.device, &mut buffer) };
        super::handle_status(status)?;
        Ok(Self { buffer, client: self.client.clone() })
    }

    /// Copy the buffer back to the host as a literal.
    pub fn to_literal_sync(&self) -> Result<Literal> {
        let mut result: c_lib::literal = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_buffer_to_literal_sync(self.buffer, &mut result) };
        super::handle_status(status)?;
        Ok(Literal(result))
    }

    /// Retrieve the shape used by this buffer.
    pub fn on_device_shape(&self) -> Result<Shape> {
        let shape = unsafe { c_lib::pjrt_buffer_on_device_shape(self.buffer) };
        let rank = unsafe { c_lib::shape_dimensions_size(shape) };
        let dimensions: Vec<_> =
            (0..rank).map(|i| unsafe { c_lib::shape_dimensions(shape, i) }).collect();
        let ty = unsafe { c_lib::shape_element_type(shape) };
        let tuple_shapes_size = unsafe { c_lib::shape_tuple_shapes_size(shape) };
        unsafe { c_lib::shape_free(shape) };
        match FromPrimitive::from_i32(ty) {
            None => Err(Error::UnexpectedElementType(ty)),
            Some(ty) => Ok(Shape { ty, dimensions, tuple_shapes_size }),
        }
    }

    /// Copy the data stored in a buffer to host memory in a blocking way.
    pub fn copy_raw_to_host_sync<T: ArrayElement>(
        &self,
        dst: &mut [T],
        offset: usize,
    ) -> Result<()> {
        let shape = self.on_device_shape()?;
        if shape.ty != T::PRIMITIVE_TYPE {
            Err(Error::ElementTypeMismatch { on_device: shape.ty, on_host: T::PRIMITIVE_TYPE })?
        }
        if offset + dst.len() > shape.element_count() {
            Err(Error::TargetBufferIsTooLarge { offset, shape, buffer_len: dst.len() })?
        }
        let status = unsafe {
            c_lib::pjrt_buffer_copy_raw_to_host_sync(
                self.buffer,
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
        unsafe { c_lib::pjrt_buffer_free(self.buffer) }
    }
}
