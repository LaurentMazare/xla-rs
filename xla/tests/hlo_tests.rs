use xla::{ArrayElement, Result};

fn build_test_computation() -> Result<xla::XlaComputation> {
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[3, 2], "x")?;
    let w = builder.parameter(1, f32::TY, &[2, 2], "w")?;
    x.matmul(&w)?.tanh()?.build()
}

#[test]
fn hlo_to_string() -> Result<()> {
    let proto = build_test_computation()?.proto();
    let hlo = proto.to_string()?;
    assert!(hlo.contains("ENTRY"), "unexpected hlo: {hlo}");
    assert!(hlo.contains("f32[3,2]"), "unexpected hlo: {hlo}");
    assert!(hlo.contains("tanh"), "unexpected hlo: {hlo}");
    Ok(())
}

#[test]
fn hlo_to_stablehlo_string() -> Result<()> {
    let proto = build_test_computation()?.proto();
    let mlir = proto.to_stablehlo_string()?;
    assert!(mlir.contains("module"), "unexpected mlir: {mlir}");
    assert!(mlir.contains("tensor<3x2xf32>"), "unexpected mlir: {mlir}");
    assert!(mlir.contains("stablehlo.tanh"), "unexpected mlir: {mlir}");
    Ok(())
}

#[test]
fn hlo_proto_roundtrip() -> Result<()> {
    let proto = build_test_computation()?.proto();
    let hlo = proto.to_string()?;

    // Text format.
    let from_text = xla::HloModuleProto::parse_and_return_unverified_module(hlo.as_bytes())?;
    assert_eq!(from_text.to_string()?, hlo);

    // Binary proto.
    let bytes = proto.to_bytes()?;
    let from_binary = xla::HloModuleProto::parse_proto(&bytes, true)?;
    assert_eq!(from_binary.to_string()?, hlo);

    // Text proto.
    let pbtxt = proto.to_pbtxt()?;
    let from_pbtxt = xla::HloModuleProto::parse_proto(pbtxt.as_bytes(), false)?;
    assert_eq!(from_pbtxt.to_string()?, hlo);

    // The roundtripped module can still be compiled and executed.
    let client = xla::PjRtClient::cpu()?;
    let exe = xla::XlaComputation::from_proto(&from_binary).compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4., 5., 6.]).reshape(&[3, 2])?;
    let w = xla::Literal::vec1(&[1f32, 0., 0., 1.]).reshape(&[2, 2])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![3, 2]));
    Ok(())
}

#[test]
fn hlo_to_file() -> Result<()> {
    let proto = build_test_computation()?.proto();
    let dir = std::env::temp_dir();
    let text_path = dir.join("xla-rs-test-hlo.txt");
    let proto_path = dir.join("xla-rs-test-hlo.pb");
    proto.to_text_file(&text_path)?;
    proto.to_proto_file(&proto_path, true)?;

    assert_eq!(xla::HloModuleProto::from_text_file(&text_path)?.to_string()?, proto.to_string()?);
    assert_eq!(
        xla::HloModuleProto::from_proto_file(&proto_path, true)?.to_string()?,
        proto.to_string()?
    );

    std::fs::remove_file(&text_path)?;
    std::fs::remove_file(&proto_path)?;
    Ok(())
}
