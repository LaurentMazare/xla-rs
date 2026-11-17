use std::ffi::c_void;
use xla::ffi::{CallFrame, DataType, Handler, HandlerTraits};
use xla::{ArrayElement, Result};

/// `out = a * x + y` on the host, `a` from a scalar attribute; checks the
/// other attributes and the buffer shapes along the way.
struct AxpyHost;

impl Handler for AxpyHost {
    fn execute(frame: &CallFrame) -> std::result::Result<(), String> {
        let (Some(x), Some(y), Some(out)) = (frame.arg(0), frame.arg(1), frame.ret(0)) else {
            return Err("axpy: expected two operands and a result".into());
        };
        let a: f32 = frame.attr("a").ok_or("axpy: missing attribute a")?;
        let n: i32 = frame.attr("n").ok_or("axpy: missing attribute n")?;
        if frame.attr::<i64>("a").is_some() || frame.attr::<f32>("missing").is_some() {
            return Err("axpy: attribute lookup ignores types or names".into());
        }
        if frame.attr_str("tag") != Some(&b"axpy-test"[..]) {
            return Err("axpy: bad string attribute".into());
        }
        if frame.num_args() != 2 || frame.num_rets() != 1 {
            return Err("axpy: bad arity".into());
        }
        let n = n as usize;
        for b in [x, y, out] {
            if b.dtype() != DataType::F32 || b.dims() != [n as i64] || b.element_count() != n {
                return Err(format!("axpy: bad buffer {:?} {:?}", b.dtype(), b.dims()));
            }
        }
        let x = unsafe { std::slice::from_raw_parts(x.data() as *const f32, n) };
        let y = unsafe { std::slice::from_raw_parts(y.data() as *const f32, n) };
        let out = unsafe { std::slice::from_raw_parts_mut(out.data() as *mut f32, n) };
        for i in 0..n {
            out[i] = a * x[i] + y[i];
        }
        Ok(())
    }
}

struct AlwaysFails;

impl Handler for AlwaysFails {
    fn execute(_: &CallFrame) -> std::result::Result<(), String> {
        Err("always_fails ran".into())
    }
}

struct Panics;

impl Handler for Panics {
    fn execute(_: &CallFrame) -> std::result::Result<(), String> {
        panic!("panics ran")
    }
}

#[test]
fn ffi_custom_call_cpu() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    xla::ffi::register_handler::<AxpyHost>("xla_rs_test_axpy", "Host")?;
    // Registering the same handler again is fine, another one under the name is not.
    xla::ffi::register_handler::<AxpyHost>("xla_rs_test_axpy", "Host")?;
    let err = xla::ffi::register_handler::<AlwaysFails>("xla_rs_test_axpy", "Host").unwrap_err();
    assert!(err.to_string().contains("xla_rs_test_axpy"), "{err}");
    let builder = xla::XlaBuilder::new("ffi");
    let x = builder.parameter(0, f32::TY, &[4], "x")?;
    let y = builder.parameter(1, f32::TY, &[4], "y")?;
    let out = builder.custom_call_ffi(
        "xla_rs_test_axpy",
        &[&x, &y],
        f32::TY,
        &[4],
        r#"{a = 2.5 : f32, n = 4 : i32, tag = "axpy-test"}"#,
    )?;
    // Something after the call, so that its result feeds the rest of the graph.
    let out = (out + builder.c0(1f32)?)?;
    let exe = out.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4.]);
    let y = xla::Literal::vec1(&[10f32, 20., 30., 40.]);
    let res = exe.execute::<xla::Literal>(&[x, y])?[0][0].to_literal_sync()?;
    assert_eq!(res.to_vec::<f32>()?, [13.5, 26., 38.5, 51.]);
    Ok(())
}

fn run_failing(client: &xla::PjRtClient, target: &str) -> String {
    let builder = xla::XlaBuilder::new("ffi-error");
    let x = builder.parameter(0, f32::TY, &[2], "x").unwrap();
    let out = builder.custom_call_ffi(target, &[&x], f32::TY, &[2], "").unwrap();
    let exe = out.build().unwrap().compile(client).unwrap();
    let Err(err) = exe
        .execute::<xla::Literal>(&[xla::Literal::vec1(&[1f32, 2.])])
        .and_then(|r| r[0][0].to_literal_sync())
    else {
        panic!("the failing custom call {target} succeeded");
    };
    err.to_string()
}

#[test]
fn ffi_custom_call_error() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    xla::ffi::register_handler::<AlwaysFails>("xla_rs_test_fails", "Host")?;
    xla::ffi::register_handler::<Panics>("xla_rs_test_panics", "Host")?;
    let err = run_failing(&client, "xla_rs_test_fails");
    assert!(err.contains("always_fails ran"), "{err}");
    let err = run_failing(&client, "xla_rs_test_panics");
    assert!(err.contains("ffi handler panicked: panics ran"), "{err}");
    Ok(())
}

#[test]
fn ffi_unknown_target() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("ffi-unknown");
    let x = builder.parameter(0, f32::TY, &[2], "x")?;
    let out = builder.custom_call_ffi("xla_rs_test_not_registered", &[&x], f32::TY, &[2], "")?;
    assert!(out.build()?.compile(&client).is_err());
    Ok(())
}

type CuMemcpyDtoDAsync = unsafe extern "C" fn(u64, u64, usize, *mut c_void) -> i32;

/// `out = x` with `cuMemcpyDtoDAsync` on the call's stream; libcuda is
/// already loaded by the gpu client.
struct CopyCuda;

impl Handler for CopyCuda {
    const TRAITS: HandlerTraits = HandlerTraits::COMMAND_BUFFER_COMPATIBLE;

    fn execute(frame: &CallFrame) -> std::result::Result<(), String> {
        let (Some(x), Some(out)) = (frame.arg(0), frame.ret(0)) else {
            return Err("copy: expected one operand and a result".into());
        };
        let stream = frame.stream().map_err(|e| e.to_string())?;
        let lib = unsafe { libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW) };
        if lib.is_null() {
            return Err("copy: libcuda.so.1 not found".into());
        }
        let f = unsafe { libc::dlsym(lib, c"cuMemcpyDtoDAsync_v2".as_ptr()) };
        if f.is_null() {
            return Err("copy: cuMemcpyDtoDAsync_v2 not found".into());
        }
        let f: CuMemcpyDtoDAsync = unsafe { std::mem::transmute(f) };
        let bytes = x.element_count() * 4;
        let r = unsafe { f(out.data() as u64, x.data() as u64, bytes, stream) };
        if r != 0 {
            return Err(format!("copy: cuMemcpyDtoDAsync failed with {r}"));
        }
        Ok(())
    }
}

#[test]
fn ffi_custom_call_gpu() -> Result<()> {
    // Runs on the GPU client when one is available, skips otherwise.
    let client = match xla::PjRtClient::gpu(0.3, false) {
        Ok(client) => client,
        Err(err) => {
            eprintln!("no gpu client, skipping ffi_custom_call_gpu: {err}");
            return Ok(());
        }
    };
    xla::ffi::register_handler::<CopyCuda>("xla_rs_test_copy", "CUDA")?;
    let builder = xla::XlaBuilder::new("ffi-gpu");
    let x = builder.parameter(0, f32::TY, &[1024], "x")?;
    // Fusions on both sides of the call, so that it sits inside a graph XLA
    // may capture as one command buffer.
    let x2 = (x * builder.c0(2f32)?)?;
    let y = builder.custom_call_ffi("xla_rs_test_copy", &[&x2], f32::TY, &[1024], "")?;
    let out = (y + builder.c0(1f32)?)?.exp()?.log()?;
    let exe = out.build()?.compile(&client)?;
    let input: Vec<f32> = (0..1024).map(|i| i as f32 / 64.).collect();
    for _ in 0..3 {
        // Replays must see the current input, not the recorded one.
        let x = xla::Literal::vec1(&input);
        let res = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?.to_vec::<f32>()?;
        for (r, v) in res.iter().zip(&input) {
            assert!((r - (2. * v + 1.)).abs() < 1e-3 * (2. * v + 1.), "{r} vs {v}");
        }
    }
    let x = xla::Literal::vec1(&vec![1f32; 1024]);
    let res = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?.to_vec::<f32>()?;
    assert!(res.iter().all(|r| (r - 3.).abs() < 1e-5));
    Ok(())
}
