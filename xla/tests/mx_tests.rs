use xla::{ArrayElement, ElementType, PrimitiveType, Result};

/// Round-trip values through a narrow type with a materialized device buffer
/// between the narrowing and widening conversions — within one computation
/// the GPU pipeline folds `convert_f32(convert_narrow(x))` back to `x` and
/// the narrow type never gets exercised (see `fp8_tests.rs`).
fn roundtrip(client: &xla::PjRtClient, ty: PrimitiveType, values: &[f32]) -> Result<Vec<f32>> {
    let n = values.len() as i64;
    let builder = xla::XlaBuilder::new("mx-narrow");
    let x = builder.parameter(0, f32::TY, &[n], "x")?;
    let x = x.convert(ty)?;
    assert_eq!(x.ty()?, ty);
    let narrow = x.build()?.compile(client)?;

    let builder = xla::XlaBuilder::new("mx-widen");
    let x = builder.parameter(0, ty.element_type()?, &[n], "x")?;
    let widen = x.convert(PrimitiveType::F32)?.build()?.compile(client)?;

    let x = xla::Literal::vec1(values);
    let narrow_buf = narrow.execute::<xla::Literal>(&[x])?.pop().unwrap().pop().unwrap();
    match narrow_buf.on_device_shape()? {
        xla::Shape::Array(a) => assert_eq!((a.ty(), a.dims()), (ty.element_type()?, &[n][..])),
        other => panic!("unexpected shape {other:?}"),
    }
    widen.execute_b(&[&narrow_buf])?[0][0].to_literal_sync()?.to_vec::<f32>()
}

fn mx_element_types() -> Result<()> {
    assert_eq!(ElementType::F8E8M0FNU.element_size_in_bytes(), 1);
    assert_eq!(ElementType::F8E8M0FNU.primitive_type(), PrimitiveType::F8E8M0FNU);
    assert_eq!(PrimitiveType::F8E8M0FNU.element_type()?, ElementType::F8E8M0FNU);
    assert_eq!(PrimitiveType::F8E8M0FNU as i32, 33);
    assert_eq!(xla::F8E8M0FNU::TY, ElementType::F8E8M0FNU);
    assert_eq!(xla::F8E8M0FNU::ELEMENT_SIZE_IN_BYTES, 1);

    assert_eq!(ElementType::F4E2M1FN.primitive_type(), PrimitiveType::F4E2M1FN);
    assert_eq!(PrimitiveType::F4E2M1FN.element_type()?, ElementType::F4E2M1FN);
    assert_eq!(PrimitiveType::F4E2M1FN as i32, 32);
    Ok(())
}

fn e8m0_roundtrip(client: &xla::PjRtClient) -> Result<()> {
    // e8m0 is exponent-only: exactly the powers of two (no zero, no sign).
    let exact = [0.0009765625f32, 0.25, 0.5, 1.0, 2.0, 4.0, 1024.0, 1048576.0];
    assert_eq!(roundtrip(client, PrimitiveType::F8E8M0FNU, &exact)?, exact);
    // A non-power-of-two lands on one of its power-of-two neighbours
    // (the exact tie-break is the backend's business).
    let out = roundtrip(client, PrimitiveType::F8E8M0FNU, &[3.0])?;
    assert!(out == [2.0] || out == [4.0], "3.0 -> {out:?}");
    Ok(())
}

fn f4_roundtrip(client: &xla::PjRtClient) -> Result<()> {
    // The full positive value set of e2m1 is {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    let exact = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -1.5, -6.0];
    assert_eq!(roundtrip(client, PrimitiveType::F4E2M1FN, &exact)?, exact);
    // Between representable points: 5.0 must land on 4 or 6.
    let out = roundtrip(client, PrimitiveType::F4E2M1FN, &[5.0])?;
    assert!(out == [4.0] || out == [6.0], "5.0 -> {out:?}");
    // Saturation-ish: above the max finite 6.0 the result stays in-set
    // (finite-only type, so either 6.0 or NaN depending on the backend's
    // overflow rule — only 6.0 round-trips to a finite value).
    Ok(())
}

fn f4_host_transfer_rejected(client: &xla::PjRtClient) -> Result<()> {
    // f4 is packed two elements per byte: the byte-per-element host transfer
    // must refuse it rather than mismatch the layout.
    let res = client.buffer_from_host_raw_bytes(ElementType::F4E2M1FN, &[0u8; 4], &[8], None);
    assert!(res.is_err(), "packed f4 host transfer should be rejected");
    Ok(())
}

#[test]
fn mx_cpu() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    mx_element_types()?;
    e8m0_roundtrip(&client)?;
    f4_roundtrip(&client)?;
    f4_host_transfer_rejected(&client)?;
    Ok(())
}

#[test]
fn mx_gpu() -> Result<()> {
    // The MX *types* are plain element types: converts and buffers do not
    // need Blackwell block-scaled hardware, only the fused block-scaled
    // matmuls do. Runs when a GPU client is available, skips otherwise.
    let client = match xla::PjRtClient::gpu(0.3, false) {
        Ok(client) => client,
        Err(err) => {
            eprintln!("no gpu client, skipping mx_gpu: {err}");
            return Ok(());
        }
    };
    e8m0_roundtrip(&client)?;
    f4_roundtrip(&client)?;
    f4_host_transfer_rejected(&client)?;
    Ok(())
}
