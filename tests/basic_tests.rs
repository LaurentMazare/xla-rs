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
fn matmul_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Batched lhs [2, 2, 3] multiplied by an unbatched rhs [3, 2], the usual
    // `x @ w` pattern. Following numpy/jax semantics, the rhs batch dimensions
    // get broadcasted and the result has shape [2, 2, 2].
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2, 2, 3], "x")?;
    let w = builder.parameter(1, f32::TY, &[3, 2], "w")?;
    let exe = x.matmul(&w)?.build()?.compile(&client)?;
    let x =
        xla::Literal::vec1(&(1..=12).map(|v| v as f32).collect::<Vec<_>>()).reshape(&[2, 2, 3])?;
    let w = xla::Literal::vec1(&[1f32, 0., 0., 1., 1., 1.]).reshape(&[3, 2])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![2, 2, 2]));
    // Each row [a, b, c] maps to [a + c, b + c].
    assert_eq!(result.to_vec::<f32>()?, [4., 5., 10., 11., 16., 17., 22., 23.]);

    // Matrix times vector, the result is squeezed to shape [2].
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2, 3], "x")?;
    let v = builder.parameter(1, f32::TY, &[3], "v")?;
    let exe = x.matmul(&v)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4., 5., 6.]).reshape(&[2, 3])?;
    let v = xla::Literal::vec1(&[1f32, 1., 1.]);
    let result = exe.execute::<xla::Literal>(&[x, v])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [6., 15.]);

    // Vector times batched matrix, numpy semantics give shape [2, 2].
    let builder = xla::XlaBuilder::new("test");
    let v = builder.parameter(0, f32::TY, &[3], "v")?;
    let x = builder.parameter(1, f32::TY, &[2, 3, 2], "x")?;
    let exe = v.matmul(&x)?.build()?.compile(&client)?;
    let v = xla::Literal::vec1(&[1f32, 1., 1.]);
    let x =
        xla::Literal::vec1(&(1..=12).map(|v| v as f32).collect::<Vec<_>>()).reshape(&[2, 3, 2])?;
    let result = exe.execute::<xla::Literal>(&[v, x])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![2, 2]));
    assert_eq!(result.to_vec::<f32>()?, [9., 12., 27., 30.]);

    // Incompatible contraction dimensions have to be rejected.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2, 3], "x")?;
    let w = builder.parameter(1, f32::TY, &[2, 4], "w")?;
    assert!(x.matmul(&w).and_then(|op| op.build()).is_err());
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
    // Tuple results get flattened into multiple output buffers.
    assert_eq!(result[0].len(), 2);
    assert_eq!(result[0][1].to_literal_sync()?.to_vec::<f32>()?, [4.2, 1.337]);
    assert_eq!(result[0][0].to_literal_sync()?.to_vec::<f32>()?, [3.1]);
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
