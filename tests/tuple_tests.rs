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
    let mut result = result[0][0].to_literal_sync()?;
    assert_eq!(result.shape()?.tuple_size(), Some(2));
    let as_tuple = result.decompose_tuple()?;
    assert_eq!(result.shape()?.tuple_size(), Some(0));
    assert_eq!(as_tuple.len(), 2);
    assert_eq!(as_tuple[0].array_shape()?, xla::ArrayShape::new::<f32>(vec![]));
    assert_eq!(as_tuple[1].array_shape()?, xla::ArrayShape::new::<f32>(vec![2]));
    assert_eq!(as_tuple[1].to_vec::<f32>()?, vec![43f32, 43f32]);
    Ok(())
}
