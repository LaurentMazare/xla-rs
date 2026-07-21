use xla::Result;

#[test]
fn tuple_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let cst42 = builder.constant_r0(42f32)?;
    let cst43 = builder.constant_r1c(43f32, 2)?;
    let computation = builder.tuple(&[cst42, cst43])?.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<xla::Literal>(&[])?;
    // Tuple results get flattened into multiple output buffers.
    assert_eq!(result[0].len(), 2);
    let as_tuple: Vec<_> = result[0].iter().map(|b| b.to_literal_sync()).collect::<Result<_>>()?;
    assert_eq!(as_tuple[0].array_shape()?, xla::ArrayShape::new::<f32>(vec![]));
    assert_eq!(as_tuple[1].array_shape()?, xla::ArrayShape::new::<f32>(vec![2]));
    assert_eq!(as_tuple[1].to_vec::<f32>()?, vec![43f32, 43f32]);
    Ok(())
}
