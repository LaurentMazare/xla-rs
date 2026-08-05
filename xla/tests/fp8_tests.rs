use xla::{ArrayElement, ElementType, PrimitiveType, Result};

/// Values are pushed through fp8 with `convert` round-trips: there is no
/// native host fp8 type, so correctness is asserted on exactly-representable
/// values (which must survive unchanged) and on the rounding of values that
/// fall between representable points.
///
/// The narrowing and widening conversions are deliberately two separate
/// executables with a real fp8 device buffer in between: within a single
/// computation the algebraic simplifier is free to fold
/// `convert_f32(convert_f8(x))` back to `x` (the GPU pipeline does), which
/// would leave the rounding unexercised. The split also covers fp8
/// parameters and fp8-typed outputs.
fn roundtrip(client: &xla::PjRtClient, ty: PrimitiveType, values: &[f32]) -> Result<Vec<f32>> {
    let n = values.len() as i64;
    let builder = xla::XlaBuilder::new("fp8-narrow");
    let x = builder.parameter(0, f32::TY, &[n], "x")?;
    let x = x.convert(ty)?;
    // Check the fp8 type is what the graph reports.
    assert_eq!(x.ty()?, ty);
    let narrow = x.build()?.compile(client)?;

    let builder = xla::XlaBuilder::new("fp8-widen");
    let x = builder.parameter(0, ty.element_type()?, &[n], "x")?;
    let widen = x.convert(PrimitiveType::F32)?.build()?.compile(client)?;

    let x = xla::Literal::vec1(values);
    let f8 = narrow.execute::<xla::Literal>(&[x])?.pop().unwrap().pop().unwrap();
    match f8.on_device_shape()? {
        xla::Shape::Array(a) => assert_eq!((a.ty(), a.dims()), (ty.element_type()?, &[n][..])),
        other => panic!("unexpected shape {other:?}"),
    }
    widen.execute_b(&[&f8])?[0][0].to_literal_sync()?.to_vec::<f32>()
}

fn fp8_element_types() -> Result<()> {
    assert_eq!(ElementType::F8E5M2.element_size_in_bytes(), 1);
    assert_eq!(ElementType::F8E4M3FN.element_size_in_bytes(), 1);
    assert_eq!(ElementType::F8E5M2.primitive_type(), PrimitiveType::F8E5M2);
    assert_eq!(ElementType::F8E4M3FN.primitive_type(), PrimitiveType::F8E4M3FN);
    assert_eq!(PrimitiveType::F8E5M2.element_type()?, ElementType::F8E5M2);
    assert_eq!(PrimitiveType::F8E4M3FN.element_type()?, ElementType::F8E4M3FN);
    assert_eq!(xla::F8E5M2::TY, ElementType::F8E5M2);
    assert_eq!(xla::F8E4M3FN::TY, ElementType::F8E4M3FN);
    assert_eq!(xla::F8E5M2::ELEMENT_SIZE_IN_BYTES, 1);
    assert_eq!(xla::F8E4M3FN::ELEMENT_SIZE_IN_BYTES, 1);
    Ok(())
}

fn fp8_convert_roundtrip(client: &xla::PjRtClient) -> Result<()> {
    // Exactly representable in e4m3fn (4-bit exponent, 3-bit mantissa,
    // max finite 448) — must survive the round-trip bit-exactly.
    let exact = [0.0f32, 0.5, -0.5, 1.0, 1.125, -2.0, 240.0, 448.0, 0.015625];
    assert_eq!(roundtrip(client, PrimitiveType::F8E4M3FN, &exact)?, exact);
    // Exactly representable in e5m2 (5-bit exponent, 2-bit mantissa,
    // max finite 57344).
    let exact = [0.0f32, 0.25, -0.75, 1.0, 1.25, 5.0, -6.0, 49152.0, 57344.0];
    assert_eq!(roundtrip(client, PrimitiveType::F8E5M2, &exact)?, exact);

    // Rounding: e4m3fn has 3 mantissa bits, so in [1, 2) the representable
    // step is 0.125 and ties go to even; e5m2 has 2 bits (step 0.25).
    let out = roundtrip(client, PrimitiveType::F8E4M3FN, &[1.05, 1.1875, 3.3])?;
    assert_eq!(out, [1.0, 1.25, 3.25]);
    let out = roundtrip(client, PrimitiveType::F8E5M2, &[1.05, 1.375, 3.3])?;
    assert_eq!(out, [1.0, 1.5, 3.5]);
    Ok(())
}

fn fp8_dot(client: &xla::PjRtClient) -> Result<()> {
    // An fp8 x fp8 -> f32 matmul in the scaled pattern the XLA gemm rewriter
    // looks for: convert both sides to fp8, convert back to a wide type,
    // multiply by scales, dot. Inputs are exactly representable so the
    // result must match the f32 reference exactly.
    let builder = xla::XlaBuilder::new("fp8-dot");
    let x = builder.parameter(0, f32::TY, &[2, 4], "x")?;
    let w = builder.parameter(1, f32::TY, &[4, 2], "w")?;
    let scale = builder.constant_r0(0.5f32)?;
    let xq = x.convert(PrimitiveType::F8E4M3FN)?.convert(PrimitiveType::F32)?;
    let wq = w.convert(PrimitiveType::F8E4M3FN)?.convert(PrimitiveType::F32)?;
    let xq = xq.mul_(&scale.broadcast(&[2, 4])?)?;
    let wq = wq.mul_(&scale.broadcast(&[4, 2])?)?;
    let exe = xq.matmul(&wq)?.build()?.compile(client)?;

    let x_host: Vec<f32> = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    let w_host: Vec<f32> = vec![1., 0., 0., 1., 1., 1., -1., -2.];
    let x = xla::Literal::vec1(&x_host).reshape(&[2, 4])?;
    let w = xla::Literal::vec1(&w_host).reshape(&[4, 2])?;
    let out = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?.to_vec::<f32>()?;

    // Reference with the same 0.25 total scaling.
    let mut expected = vec![0f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..4 {
                expected[i * 2 + j] += 0.25 * x_host[i * 4 + k] * w_host[k * 2 + j];
            }
        }
    }
    assert_eq!(out, expected);
    Ok(())
}

fn fp8_buffer_shape(client: &xla::PjRtClient) -> Result<()> {
    // Device buffers can hold fp8 directly; the on-device shape reports the
    // fp8 element type.
    let builder = xla::XlaBuilder::new("fp8-shape");
    let x = builder.parameter(0, f32::TY, &[8], "x")?;
    let exe = x.convert(PrimitiveType::F8E4M3FN)?.build()?.compile(client)?;
    let x = xla::Literal::vec1(&[0f32, 1., 2., 3., 4., 5., 6., 7.]);
    let out = &exe.execute::<xla::Literal>(&[x])?[0][0];
    let shape = out.on_device_shape()?;
    match shape {
        xla::Shape::Array(a) => {
            assert_eq!(a.ty(), ElementType::F8E4M3FN);
            assert_eq!(a.dims(), [8]);
        }
        other => panic!("unexpected shape {other:?}"),
    };
    Ok(())
}

#[test]
fn fp8_cpu() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    fp8_element_types()?;
    fp8_convert_roundtrip(&client)?;
    fp8_dot(&client)?;
    fp8_buffer_shape(&client)?;
    Ok(())
}

#[test]
fn fp8_gpu() -> Result<()> {
    // Runs on the GPU client when one is available (needs a CUDA
    // xla_extension build and a visible device), skips otherwise so plain
    // CPU CI stays green.
    let client = match xla::PjRtClient::gpu(0.3, false) {
        Ok(client) => client,
        Err(err) => {
            eprintln!("no gpu client, skipping fp8_gpu: {err}");
            return Ok(());
        }
    };
    fp8_convert_roundtrip(&client)?;
    fp8_dot(&client)?;
    fp8_buffer_shape(&client)?;
    Ok(())
}
