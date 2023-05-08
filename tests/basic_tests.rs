use xla::{ArrayElement, Result};

#[test]
fn add_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let cst42 = builder.constant_r0(42f32)?;
    let cst43 = builder.constant_r1c(43f32, 2)?;
    let sum = (cst42 + &cst43)?;
    let computation = sum.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.element_count(), 2);
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![2]));
    assert_eq!(result.get_first_element::<f32>()?, 85.);
    assert_eq!(result.to_vec::<f32>()?, [85., 85.]);
    Ok(())
}

#[test]
fn sum_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2], "x")?;
    let sum = x.reduce_sum(&[], false)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [4.2, 1.337]);

    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-2], "x")?;
    let sum = x.reduce_sum(&[0], false)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [5.5369997]);
    // Dimensions got reduced.
    assert_eq!(result.array_shape()?.dims(), []);

    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-2], "x")?;
    let sum = x.reduce_sum(&[0], true)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [5.5369997]);
    // keep_dims = true in this case.
    assert_eq!(result.array_shape()?.dims(), [1]);
    Ok(())
}

#[test]
fn mean_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-2], "x")?;
    let sum = x.reduce_mean(&[0], false)?.build()?.compile(&client)?;
    let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = sum.execute::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [2.7684999]);
    // Dimensions got reduced.
    assert_eq!(result.array_shape()?.dims(), []);
    Ok(())
}

#[test]
fn tuple_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[-1], "x")?;
    let y = builder.parameter(1, f32::TY, &[2], "x")?;
    let tuple = builder.tuple(&[x, y])?.build()?.compile(&client)?;
    let x = xla::Literal::scalar(3.1f32);
    let y = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = tuple.execute::<xla::Literal>(&[x, y])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.shape()?.tuple_size(), Some(2));
    let mut result = result;
    let result = result.decompose_tuple()?;
    assert_eq!(result[1].to_vec::<f32>()?, [4.2, 1.337]);
    assert_eq!(result[0].to_vec::<f32>()?, [3.1]);
    Ok(())
}

#[test]
fn tuple_literal() -> Result<()> {
    let x = xla::Literal::scalar(3.1f32);
    let y = xla::Literal::vec1(&[4.2f32, 1.337f32]);
    let result = xla::Literal::tuple(vec![x, y]);
    assert_eq!(result.shape()?.tuple_size(), Some(2));
    let mut result = result;
    let result = result.decompose_tuple()?;
    assert_eq!(result[1].to_vec::<f32>()?, [4.2, 1.337]);
    assert_eq!(result[0].to_vec::<f32>()?, [3.1]);
    Ok(())
}
