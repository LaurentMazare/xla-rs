//! A device (CPUs, GPUs, TPUs) where computations can be run.
use super::{ArrayElement, Literal, PjRtBuffer, PjRtDevice, PjRtLoadedExecutable, XlaComputation};
use crate::{c_lib, Error, Result};
use std::marker::PhantomData;
use std::sync::Arc;

pub(super) struct PjRtClientInternal(pub(self) c_lib::pjrt_client);

// SAFETY: the PJRT C++ API is thread-safe: clients, buffers, and executables
// can be used concurrently from multiple threads. The internal struct only
// wraps the raw client pointer.
unsafe impl Send for PjRtClientInternal {}
unsafe impl Sync for PjRtClientInternal {}

/// A value in the options handed to a PJRT plugin when creating a client, see
/// [`PjRtClient::plugin_with_options`].
#[derive(Debug, Clone, PartialEq)]
pub enum PluginOption {
    Str(String),
    Bool(bool),
    Int(i64),
    Float(f32),
    IntList(Vec<i64>),
}

impl PluginOption {
    /// The type tag understood by `pjrt_plugin_client_create`.
    fn tag(&self) -> i32 {
        match self {
            Self::Str(_) => 0,
            Self::Bool(_) => 1,
            Self::Int(_) => 2,
            Self::Float(_) => 3,
            Self::IntList(_) => 4,
        }
    }

    /// The string encoding decoded on the C++ side according to [`Self::tag`].
    fn encode(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Bool(b) => b.to_string(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{f:?}"),
            Self::IntList(is) => is.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
        }
    }
}

/// A client represents a device that can be used to run some computations. A computation graph is
/// compiled in a way that is specific to a device before it can be run.
#[derive(Clone)]
pub struct PjRtClient(Arc<PjRtClientInternal>);

impl PjRtClient {
    /// A CPU client, this can run computations on multiple CPUs at the same time.
    pub fn cpu() -> Result<Self> {
        let mut ptr: c_lib::pjrt_client = std::ptr::null_mut();
        let status = unsafe { c_lib::pjrt_cpu_client_create(&mut ptr) };
        super::handle_status(status)?;
        Ok(Self(Arc::new(PjRtClientInternal(ptr))))
    }

    /// A GPU client, the memory requirements are limited by the specified `memory_fraction` and
    /// this memory can either be allocated dynamically or pre-allocated depending on
    /// `preallocate`.
    pub fn gpu(memory_fraction: f64, preallocate: bool) -> Result<Self> {
        let mut ptr: c_lib::pjrt_client = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_gpu_client_create(&mut ptr, memory_fraction, preallocate) };
        super::handle_status(status)?;
        Ok(Self(Arc::new(PjRtClientInternal(ptr))))
    }

    /// A client on a PJRT C-API plugin: `library_path` is a shared library
    /// exporting `GetPjrtApi` (e.g. jax's `xla_cuda_plugin.so`, `libtpu.so`,
    /// or `libneuronpjrt.so`), and `device_type` the name it is registered
    /// under in the process-wide plugin registry (case insensitive). A second
    /// client for an already loaded plugin reuses the library.
    pub fn plugin<P: AsRef<std::path::Path>>(device_type: &str, library_path: P) -> Result<Self> {
        Self::plugin_with_options(device_type, library_path, &[])
    }

    /// Same as [`Self::plugin`], with the plugin's client creation options,
    /// e.g. `("memory_fraction", PluginOption::Float(0.5))` for the cuda one.
    pub fn plugin_with_options<P: AsRef<std::path::Path>>(
        device_type: &str,
        library_path: P,
        options: &[(&str, PluginOption)],
    ) -> Result<Self> {
        use std::ffi::CString;
        let device_type = CString::new(device_type)?;
        let library_path = CString::new(library_path.as_ref().to_string_lossy().as_bytes())?;
        type Strings = std::result::Result<Vec<CString>, std::ffi::NulError>;
        let keys = options.iter().map(|(k, _)| CString::new(*k)).collect::<Strings>()?;
        let values = options.iter().map(|(_, v)| CString::new(v.encode())).collect::<Strings>()?;
        let tags: Vec<i32> = options.iter().map(|(_, v)| v.tag()).collect();
        let key_ptrs: Vec<*const std::ffi::c_char> = keys.iter().map(|k| k.as_ptr()).collect();
        let value_ptrs: Vec<*const std::ffi::c_char> = values.iter().map(|v| v.as_ptr()).collect();
        let mut ptr: c_lib::pjrt_client = std::ptr::null_mut();
        let status = unsafe {
            c_lib::pjrt_plugin_client_create(
                &mut ptr,
                device_type.as_ptr(),
                library_path.as_ptr(),
                options.len() as i32,
                key_ptrs.as_ptr(),
                tags.as_ptr(),
                value_ptrs.as_ptr(),
            )
        };
        super::handle_status(status)?;
        Ok(Self(Arc::new(PjRtClientInternal(ptr))))
    }

    /// A TPU client.
    pub fn tpu(max_inflight_computations: usize) -> Result<Self> {
        let mut ptr: c_lib::pjrt_client = std::ptr::null_mut();
        let status =
            unsafe { c_lib::pjrt_tpu_client_create(&mut ptr, max_inflight_computations as i32) };
        super::handle_status(status)?;
        Ok(Self(Arc::new(PjRtClientInternal(ptr))))
    }

    /// A client for the best available platform: try TPU, then GPU, then fall
    /// back to CPU. When `force_cpu` is set, a CPU client is created directly.
    ///
    /// The client constructors return an error when their runtime is missing
    /// (e.g. no `libtpu.so`, or no CUDA device), so the same binary works
    /// against the cpu, cuda, or tpu `xla_extension` builds.
    pub fn auto(force_cpu: bool) -> Result<Self> {
        if force_cpu {
            return Self::cpu();
        }
        if let Ok(client) = Self::tpu(1) {
            return Ok(client);
        }
        if let Ok(client) = Self::gpu(0.90, false) {
            return Ok(client);
        }
        Self::cpu()
    }

    fn ptr(&self) -> c_lib::pjrt_client {
        self.0 .0
    }

    /// Compile a computation for this device, and return the executable.
    pub fn compile(&self, c: &XlaComputation) -> Result<PjRtLoadedExecutable> {
        let mut exe: c_lib::pjrt_loaded_executable = std::ptr::null_mut();
        let status = unsafe { c_lib::compile(self.ptr(), c.0, &mut exe) };
        super::handle_status(status)?;
        Ok(PjRtLoadedExecutable { exe, client: self.clone() })
    }

    /// Compile a computation with the gpu gemm autotuner results pinned to a
    /// file: `load_from` reuses previously dumped results, making the kernel
    /// selection deterministic and skipping the tuning, `dump_to` writes the
    /// results of this compilation (cumulated with previously tuned
    /// computations). These map to the `xla_gpu_load_autotune_results_from`
    /// and `xla_gpu_dump_autotune_results_to` debug options, scoped to this
    /// compilation rather than set process-wide through `XLA_FLAGS`.
    pub fn compile_with_autotune_cache(
        &self,
        c: &XlaComputation,
        load_from: Option<&str>,
        dump_to: Option<&str>,
    ) -> Result<PjRtLoadedExecutable> {
        let load_from = load_from.map(|s| std::ffi::CString::new(s).unwrap());
        let dump_to = dump_to.map(|s| std::ffi::CString::new(s).unwrap());
        let as_ptr =
            |s: &Option<std::ffi::CString>| s.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
        let mut exe: c_lib::pjrt_loaded_executable = std::ptr::null_mut();
        let status = unsafe {
            c_lib::compile_with_autotune_cache(
                self.ptr(),
                c.0,
                as_ptr(&load_from),
                as_ptr(&dump_to),
                &mut exe,
            )
        };
        super::handle_status(status)?;
        Ok(PjRtLoadedExecutable { exe, client: self.clone() })
    }

    /// The number of devices that this client has detected, e.g. the number of GPUs.
    pub fn device_count(&self) -> usize {
        unsafe { c_lib::pjrt_client_device_count(self.ptr()) as usize }
    }

    /// The number of devices that this client can use.
    pub fn addressable_device_count(&self) -> usize {
        unsafe { c_lib::pjrt_client_addressable_device_count(self.ptr()) as usize }
    }

    /// The name of the platform.
    pub fn platform_name(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_client_platform_name(self.ptr());
            super::c_ptr_to_string(ptr)
        }
    }

    /// The version of the platform.
    pub fn platform_version(&self) -> String {
        unsafe {
            let ptr = c_lib::pjrt_client_platform_version(self.ptr());
            super::c_ptr_to_string(ptr)
        }
    }

    /// A list of devices attached to this client.
    pub fn devices(&self) -> Vec<PjRtDevice<'_>> {
        let device_count = self.device_count();
        let mut device_ptrs = vec![std::ptr::null_mut(); device_count];
        unsafe { c_lib::pjrt_client_devices(self.ptr(), device_ptrs.as_mut_ptr()) };
        device_ptrs.into_iter().map(|device| PjRtDevice { device, marker: PhantomData }).collect()
    }

    /// A list of devices that can be used by this client.
    pub fn addressable_devices(&self) -> Vec<PjRtDevice<'_>> {
        let device_count = self.addressable_device_count();
        let mut device_ptrs = vec![std::ptr::null_mut(); device_count];
        unsafe { c_lib::pjrt_client_addressable_devices(self.ptr(), device_ptrs.as_mut_ptr()) };
        device_ptrs.into_iter().map(|device| PjRtDevice { device, marker: PhantomData }).collect()
    }

    /// Transfer some data from the host to a `PjRtBuffer` stored on the target device. If the
    /// device is not specified, the default device is used.
    /// The source data is passed as a slice of the specified primitive type, as well as the
    /// dimensions. The dimensions have to match the number of elements in the source data,
    /// otherwise an error is returned.
    pub fn buffer_from_host_buffer<T: ArrayElement>(
        &self,
        data: &[T],
        dims: &[usize],
        device: Option<&PjRtDevice>,
    ) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let element_count: usize = dims.iter().product();
        if element_count != data.len() {
            Err(Error::WrongElementCount { dims: dims.to_vec(), element_count })?
        }
        let device = device.map_or(std::ptr::null_mut(), |d| d.device);
        let dims: Vec<_> = dims.iter().map(|d| *d as i64).collect();
        let status = unsafe {
            c_lib::pjrt_buffer_from_host_buffer(
                self.ptr(),
                device,
                data.as_ptr() as *const libc::c_void,
                T::TY.primitive_type() as i32,
                dims.len() as i32,
                dims.as_ptr(),
                &mut buffer,
            )
        };
        super::handle_status(status)?;
        Ok(PjRtBuffer { buffer, client: self.clone() })
    }

    /// Transfer some data from the host to a `PjRtBuffer` stored on the target device. If the
    /// device is not specified, the default device is used.
    /// The source data is passed as a slice of raw bytes, as well as the dimensions. The
    /// dimensions have to match the number of bytes in the source data, otherwise an error
    /// is returned.
    pub fn buffer_from_host_raw_bytes(
        &self,
        ty: super::ElementType,
        data: &[u8],
        dims: &[usize],
        device: Option<&PjRtDevice>,
    ) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        // Sub-byte types are stored packed (two f4 elements per byte), so the
        // per-element byte accounting below would silently mismatch the
        // layout PJRT expects.
        if ty == super::ElementType::F4E2M1FN {
            Err(Error::UnexpectedElementType(super::PrimitiveType::F4E2M1FN as i32))?
        }
        let element_count: usize = dims.iter().product();
        let element_size_in_bytes = ty.element_size_in_bytes();
        if element_count * element_size_in_bytes != data.len() {
            Err(Error::WrongElementCount { dims: dims.to_vec(), element_count })?
        }
        let device = device.map_or(std::ptr::null_mut(), |d| d.device);
        let dims: Vec<_> = dims.iter().map(|d| *d as i64).collect();
        let status = unsafe {
            c_lib::pjrt_buffer_from_host_buffer(
                self.ptr(),
                device,
                data.as_ptr() as *const libc::c_void,
                ty.primitive_type() as i32,
                dims.len() as i32,
                dims.as_ptr(),
                &mut buffer,
            )
        };
        super::handle_status(status)?;
        Ok(PjRtBuffer { buffer, client: self.clone() })
    }

    /// Transfer some data from the host to a `PjRtBuffer` stored on the target device. If the
    /// device is not specified, the default device is used.
    /// The source data is passed as a literal.
    pub fn buffer_from_host_literal(
        &self,
        device: Option<&PjRtDevice>,
        literal: &Literal,
    ) -> Result<PjRtBuffer> {
        let mut buffer: c_lib::pjrt_buffer = std::ptr::null_mut();
        let device = device.map_or(std::ptr::null_mut(), |d| d.device);
        let status = unsafe {
            c_lib::pjrt_buffer_from_host_literal(self.ptr(), device, literal.0, &mut buffer)
        };
        super::handle_status(status)?;
        Ok(PjRtBuffer { buffer, client: self.clone() })
    }
}

impl Drop for PjRtClientInternal {
    fn drop(&mut self) {
        unsafe { c_lib::pjrt_client_free(self.0) }
    }
}
