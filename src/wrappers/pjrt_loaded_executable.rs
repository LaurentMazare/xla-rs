use super::{Literal, PjRtBuffer};
use crate::{c_lib, Result};
use std::marker::PhantomData;

pub struct PjRtLoadedExecutable<'a> {
    pub(super) exe: c_lib::pjrt_loaded_executable,
    pub(super) marker: PhantomData<&'a super::PjRtClient>,
}

impl<'a> PjRtLoadedExecutable<'a> {
    fn process_execute_outputs(outputs: *mut *mut c_lib::pjrt_buffer) -> Vec<Vec<PjRtBuffer<'a>>> {
        unsafe {
            let mut vec = vec![];
            loop {
                let outputs = *outputs.add(vec.len());
                if outputs.is_null() {
                    break;
                }
                let mut replica_vec = vec![];
                loop {
                    let buffer = *outputs.add(replica_vec.len());
                    if buffer.is_null() {
                        break;
                    }
                    replica_vec.push(PjRtBuffer { buffer, marker: PhantomData });
                }
                libc::free(outputs as *mut libc::c_void);
                vec.push(replica_vec);
            }
            libc::free(outputs as *mut libc::c_void);
            vec
        }
    }

    pub fn execute<L: std::borrow::Borrow<Literal>>(
        &self,
        args: &[L],
    ) -> Result<Vec<Vec<PjRtBuffer<'a>>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.borrow().0).collect();
        let status =
            unsafe { c_lib::execute(self.exe, args.as_ptr(), args.len() as i32, &mut outputs) };
        super::handle_status(status)?;
        Ok(Self::process_execute_outputs(outputs))
    }

    pub fn execute_b<L: std::borrow::Borrow<PjRtBuffer<'a>>>(
        &self,
        args: &[L],
    ) -> Result<Vec<Vec<PjRtBuffer<'a>>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.borrow().buffer).collect();
        let status =
            unsafe { c_lib::execute_b(self.exe, args.as_ptr(), args.len() as i32, &mut outputs) };
        super::handle_status(status)?;
        Ok(Self::process_execute_outputs(outputs))
    }
}

impl Drop for PjRtLoadedExecutable<'_> {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_loaded_executable_free(self.exe) }
    }
}
