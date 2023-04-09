use super::{Literal, PjRtBuffer};
use crate::{c_lib, Result};

pub struct PjRtLoadedExecutable(pub(super) c_lib::pjrt_loaded_executable);

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

    pub fn execute<P: std::borrow::Borrow<PjRtBuffer>>(
        &self,
        args: &[P],
    ) -> Result<Vec<Vec<PjRtBuffer>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.borrow().0).collect();
        let status =
            unsafe { c_lib::execute(self.0, args.as_ptr(), args.len() as i32, &mut outputs) };
        super::handle_status(status)?;
        Ok(Self::process_execute_outputs(outputs))
    }

    pub fn execute_literal<L: std::borrow::Borrow<Literal>>(
        &self,
        args: &[L],
    ) -> Result<Vec<Vec<PjRtBuffer>>> {
        let mut outputs = std::ptr::null_mut();
        let args: Vec<_> = args.iter().map(|x| x.borrow().0).collect();
        let status = unsafe {
            c_lib::execute_literal(self.0, args.as_ptr(), args.len() as i32, &mut outputs)
        };
        super::handle_status(status)?;
        Ok(Self::process_execute_outputs(outputs))
    }
}

impl Drop for PjRtLoadedExecutable {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_loaded_executable_free(self.0) }
    }
}
