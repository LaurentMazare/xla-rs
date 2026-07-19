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
fn pad_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Pad a [2, 2] matrix with one row at the top and one column on the right.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2, 2], "x")?;
    let zero = builder.zero(xla::ElementType::F32)?;
    let exe = x.pad(&zero, &[(1, 0, 0), (0, 1, 0)])?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4.]).reshape(&[2, 2])?;
    let result = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![3, 3]));
    assert_eq!(result.to_vec::<f32>()?, [0., 0., 0., 1., 2., 0., 3., 4., 0.]);

    // Interior padding inserts values between the original elements.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[3], "x")?;
    let zero = builder.zero(xla::ElementType::F32)?;
    let exe = x.pad(&zero, &[(0, 0, 1)])?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3.]);
    let result = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [1., 0., 2., 0., 3.]);

    // Single dimension padding with a negative dimension index.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2, 2], "x")?;
    let zero = builder.zero(xla::ElementType::F32)?;
    let exe = x.pad_in_dim(&zero, -1, 1, 1)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4.]).reshape(&[2, 2])?;
    let result = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![2, 4]));
    assert_eq!(result.to_vec::<f32>()?, [0., 1., 2., 0., 0., 3., 4., 0.]);
    Ok(())
}

#[test]
fn scatter_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Scatter-add the updates [1, 2] at positions 0 and 2 of a [4] vector.
    let add = {
        let builder = xla::XlaBuilder::new("add");
        let x = builder.parameter(0, f32::TY, &[], "x")?;
        let y = builder.parameter(1, f32::TY, &[], "y")?;
        (x + y)?.build()?
    };
    let builder = xla::XlaBuilder::new("test");
    let operand = builder.parameter(0, f32::TY, &[4], "operand")?;
    let indices = builder.parameter(1, i64::TY, &[2], "indices")?;
    let updates = builder.parameter(2, f32::TY, &[2], "updates")?;
    let exe =
        operand.scatter(&indices, &updates, &add, &[], &[0], &[0], 1)?.build()?.compile(&client)?;
    let operand = xla::Literal::vec1(&[10f32, 20., 30., 40.]);
    let indices = xla::Literal::vec1(&[0i64, 2]);
    let updates = xla::Literal::vec1(&[1f32, 2.]);
    let result =
        exe.execute::<xla::Literal>(&[operand, indices, updates])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [11., 20., 32., 40.]);

    // Scatter rows [2] at positions 2 and 0 of a [3, 2] matrix, replacing the
    // original values.
    let replace = {
        let builder = xla::XlaBuilder::new("replace");
        let _x = builder.parameter(0, f32::TY, &[], "x")?;
        let y = builder.parameter(1, f32::TY, &[], "y")?;
        // The update computation only keeps the update value.
        y.build()?
    };
    let builder = xla::XlaBuilder::new("test");
    let operand = builder.parameter(0, f32::TY, &[3, 2], "operand")?;
    let indices = builder.parameter(1, i64::TY, &[2], "indices")?;
    let updates = builder.parameter(2, f32::TY, &[2, 2], "updates")?;
    let exe = operand
        .scatter(&indices, &updates, &replace, &[1], &[0], &[0], 1)?
        .build()?
        .compile(&client)?;
    let operand = xla::Literal::vec1(&[0f32; 6]).reshape(&[3, 2])?;
    let indices = xla::Literal::vec1(&[2i64, 0]);
    let updates = xla::Literal::vec1(&[1f32, 2., 3., 4.]).reshape(&[2, 2])?;
    let result =
        exe.execute::<xla::Literal>(&[operand, indices, updates])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [3., 4., 0., 0., 1., 2.]);
    Ok(())
}

#[test]
fn sort_ops() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Ascending sort on the last dimension.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[4], "x")?;
    let exe = x.sort_asc(-1, false)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[3f32, 1., 4., 2.]);
    let result = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [1., 2., 3., 4.]);

    // Descending sort along the first dimension of a [2, 2] matrix.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2, 2], "x")?;
    let exe = x.sort_desc(0, false)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 4., 3., 2.]).reshape(&[2, 2])?;
    let result = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [3., 4., 1., 2.]);

    // The largest three values together with their indexes.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[5], "x")?;
    let exe = x.top_k(3, true)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[3f32, 1., 4., 1., 5.]);
    let result = exe.execute::<xla::Literal>(&[x])?;
    // The tuple result gets flattened into a values and an indexes buffer.
    assert_eq!(result[0].len(), 2);
    assert_eq!(result[0][0].to_literal_sync()?.to_vec::<f32>()?, [5., 4., 3.]);
    assert_eq!(result[0][1].to_literal_sync()?.to_vec::<i32>()?, [4, 2, 0]);

    // Indexes of the min/max values on the last dimension.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[2, 3], "x")?;
    let argmax = x.argmax(xla::ElementType::S32, -1)?;
    let argmin = x.argmin(xla::ElementType::S32, -1)?;
    let exe = builder.tuple(&[argmax, argmin])?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 5., 2., 7., 3., 4.]).reshape(&[2, 3])?;
    let result = exe.execute::<xla::Literal>(&[x])?;
    assert_eq!(result[0][0].to_literal_sync()?.to_vec::<i32>()?, [1, 0]);
    assert_eq!(result[0][1].to_literal_sync()?.to_vec::<i32>()?, [0, 1]);
    Ok(())
}

#[test]
fn dynamic_slice_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Slice two values at a runtime provided offset.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[5], "x")?;
    let start = builder.parameter(1, i64::TY, &[], "start")?;
    let exe = x.dynamic_slice(&[start], &[2])?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[10f32, 11., 12., 13., 14.]);
    let start = xla::Literal::scalar(2i64);
    let result = exe.execute::<xla::Literal>(&[x, start])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [12., 13.]);

    // Out of bound start indices get clamped.
    let x = xla::Literal::vec1(&[10f32, 11., 12., 13., 14.]);
    let start = xla::Literal::scalar(4i64);
    let result = exe.execute::<xla::Literal>(&[x, start])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [13., 14.]);

    // A [2, 2] slice of a [3, 3] matrix at offset (1, 1).
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[3, 3], "x")?;
    let one = builder.c0(1i64)?;
    let exe = x.dynamic_slice(&[&one, &one], &[2, 2])?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&(0..9).map(|v| v as f32).collect::<Vec<_>>()).reshape(&[3, 3])?;
    let result = exe.execute::<xla::Literal>(&[x])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [4., 5., 7., 8.]);
    Ok(())
}

#[test]
fn dynamic_update_slice_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Overwrite two values at a runtime provided offset, the kv-cache update pattern.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[5], "x")?;
    let update = builder.parameter(1, f32::TY, &[2], "update")?;
    let start = builder.parameter(2, i64::TY, &[], "start")?;
    let exe = x.dynamic_update_slice(&update, &[start])?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[0f32, 1., 2., 3., 4.]);
    let update = xla::Literal::vec1(&[20f32, 21.]);
    let start = xla::Literal::scalar(2i64);
    let result = exe.execute::<xla::Literal>(&[x, update, start])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [0., 1., 20., 21., 4.]);
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
