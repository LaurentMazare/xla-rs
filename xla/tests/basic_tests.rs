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
fn conv1d_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Plain conv1d: input [1, 1, 5], kernel [1, 1, 3] of ones, stride 1.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1, 1, 5], "x")?;
    let w = builder.parameter(1, f32::TY, &[1, 1, 3], "w")?;
    let exe = x.conv1d(&w, 1, 0, 1, 1)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4., 5.]).reshape(&[1, 1, 5])?;
    let w = xla::Literal::vec1(&[1f32, 1., 1.]).reshape(&[1, 1, 3])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![1, 1, 3]));
    // Sliding window sums: 1+2+3, 2+3+4, 3+4+5.
    assert_eq!(result.to_vec::<f32>()?, [6., 9., 12.]);

    // Strided conv1d: stride 2 keeps windows starting at positions 0 and 2.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1, 1, 5], "x")?;
    let w = builder.parameter(1, f32::TY, &[1, 1, 3], "w")?;
    let exe = x.conv1d(&w, 2, 0, 1, 1)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4., 5.]).reshape(&[1, 1, 5])?;
    let w = xla::Literal::vec1(&[1f32, 1., 1.]).reshape(&[1, 1, 3])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [6., 12.]);

    // Depthwise conv1d: 2 channels, groups = 2, kernel [2, 1, 2].
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1, 2, 3], "x")?;
    let w = builder.parameter(1, f32::TY, &[2, 1, 2], "w")?;
    let exe = x.conv1d(&w, 1, 0, 1, 2)?.build()?.compile(&client)?;
    // channel 0: [1, 2, 3], channel 1: [4, 5, 6].
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4., 5., 6.]).reshape(&[1, 2, 3])?;
    // channel 0 kernel [1, 1], channel 1 kernel [1, -1].
    let w = xla::Literal::vec1(&[1f32, 1., 1., -1.]).reshape(&[2, 1, 2])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![1, 2, 2]));
    // ch0: 1+2, 2+3 ; ch1: 4-5, 5-6.
    assert_eq!(result.to_vec::<f32>()?, [3., 5., -1., -1.]);
    Ok(())
}

#[test]
fn conv_transpose1d_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;

    // Transposed conv, stride 2, kernel [1, 1, 2] of ones. Each input value is
    // spread over `stride` output positions: output length (3-1)*2 + 2 = 6.
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1, 1, 3], "x")?;
    let w = builder.parameter(1, f32::TY, &[1, 1, 2], "w")?;
    let exe = x.conv_transpose1d(&w, 2, 0, 0, 1, 1)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[1f32, 2., 3.]).reshape(&[1, 1, 3])?;
    let w = xla::Literal::vec1(&[1f32, 1.]).reshape(&[1, 1, 2])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![1, 1, 6]));
    assert_eq!(result.to_vec::<f32>()?, [1., 1., 2., 2., 3., 3.]);

    // Depthwise transposed conv: 2 channels, groups = 2, kernel [2, 1, 2].
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1, 2, 2], "x")?;
    let w = builder.parameter(1, f32::TY, &[2, 1, 2], "w")?;
    let exe = x.conv_transpose1d(&w, 2, 0, 0, 1, 2)?.build()?.compile(&client)?;
    // ch0: [1, 2], ch1: [3, 4].
    let x = xla::Literal::vec1(&[1f32, 2., 3., 4.]).reshape(&[1, 2, 2])?;
    // ch0 kernel [1, 1], ch1 kernel [1, 2].
    let w = xla::Literal::vec1(&[1f32, 1., 1., 2.]).reshape(&[2, 1, 2])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![1, 2, 4]));
    // ch0: 1,1,2,2 ; ch1: 3,6,4,8.
    assert_eq!(result.to_vec::<f32>()?, [1., 1., 2., 2., 3., 6., 4., 8.]);

    // groups=1, 2 in-channels, 2 out-channels, stride 1, to check the
    // input/output channel mapping (not just depthwise). weight[i, o, :].
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1, 2, 2], "x")?;
    let w = builder.parameter(1, f32::TY, &[2, 2, 2], "w")?;
    let exe = x.conv_transpose1d(&w, 1, 0, 0, 1, 1)?.build()?.compile(&client)?;
    // in ch0 = [1, 0] (impulse), in ch1 = [0, 1].
    let x = xla::Literal::vec1(&[1f32, 0., 0., 1.]).reshape(&[1, 2, 2])?;
    // w[0,0]=[1,0], w[0,1]=[0,0], w[1,0]=[0,0], w[1,1]=[1,0].
    let w = xla::Literal::vec1(&[1f32, 0., 0., 0., 0., 0., 1., 0.]).reshape(&[2, 2, 2])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![1, 2, 3]));
    // out ch0 gets in ch0 via w[0,0]: [1,0,0]; out ch1 gets in ch1 via w[1,1]: [0,1,0].
    assert_eq!(result.to_vec::<f32>()?, [1., 0., 0., 0., 1., 0.]);

    // stride 2 with k=4 (overlap-add): each output position receives from two
    // input elements. input [a, b], weight [w0, w1, w2, w3].
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1, 1, 2], "x")?;
    let w = builder.parameter(1, f32::TY, &[1, 1, 4], "w")?;
    let exe = x.conv_transpose1d(&w, 2, 0, 0, 1, 1)?.build()?.compile(&client)?;
    let x = xla::Literal::vec1(&[2f32, 3.]).reshape(&[1, 1, 2])?;
    let w = xla::Literal::vec1(&[1f32, 10., 100., 1000.]).reshape(&[1, 1, 4])?;
    let result = exe.execute::<xla::Literal>(&[x, w])?[0][0].to_literal_sync()?;
    assert_eq!(result.array_shape()?, xla::ArrayShape::new::<f32>(vec![1, 1, 6]));
    // a*[1,10,100,1000] at 0.. + b*[1,10,100,1000] at 2..
    // = [2, 20, 200+3, 2000+30, 300, 3000] = [2, 20, 203, 2030, 300, 3000].
    assert_eq!(result.to_vec::<f32>()?, [2., 20., 203., 2030., 300., 3000.]);
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

#[test]
fn rng_bit_generator_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let state = builder.parameter(0, u64::TY, &[2], "state")?;
    let out =
        state.rng_bit_generator(xla::RandomAlgorithm::ThreeFry, xla::ElementType::U32, &[8])?;
    let new_state = out.get_tuple_element(0)?;
    let bits = out.get_tuple_element(1)?;
    let exe = builder.tuple(&[&new_state, &bits])?.build()?.compile(&client)?;
    let seed = xla::Literal::vec1(&[42u64, 0u64]);
    // The output tuple is flattened into one buffer per element.
    let r1 = &exe.execute::<xla::Literal>(std::slice::from_ref(&seed))?[0];
    let s1 = r1[0].to_literal_sync()?.to_vec::<u64>()?;
    let b1 = r1[1].to_literal_sync()?.to_vec::<u32>()?;
    assert_eq!(b1.len(), 8);
    // Deterministic: the same state yields the same bits.
    let r2 = &exe.execute::<xla::Literal>(&[seed])?[0];
    assert_eq!(r2[1].to_literal_sync()?.to_vec::<u32>()?, b1);
    // The returned state differs and yields a different stream.
    assert_ne!(s1, vec![42u64, 0u64]);
    let seed2 = xla::Literal::vec1(&s1);
    let r3 = &exe.execute::<xla::Literal>(&[seed2])?[0];
    assert_ne!(r3[1].to_literal_sync()?.to_vec::<u32>()?, b1);
    Ok(())
}

#[test]
fn approx_top_k_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let xs = builder.parameter(0, f32::TY, &[8], "xs")?;
    let out = xs.approx_top_k(3, -1, 0.99)?;
    let values = out.get_tuple_element(0)?;
    let indices = out.get_tuple_element(1)?;
    let exe = builder.tuple(&[&values, &indices])?.build()?.compile(&client)?;
    let xs = xla::Literal::vec1(&[1f32, 7., 3., 9., 5., 2., 8., 4.]);
    let r = &exe.execute::<xla::Literal>(&[xs])?[0];
    let mut values = r[0].to_literal_sync()?.to_vec::<f32>()?;
    let mut indices = r[1].to_literal_sync()?.to_vec::<i32>()?;
    values.sort_by(|a, b| b.partial_cmp(a).unwrap());
    indices.sort_unstable();
    // Small inputs aggregate to the exact top-k.
    assert_eq!(values, [9., 8., 7.]);
    assert_eq!(indices, [1, 3, 6]);
    Ok(())
}

#[test]
fn sample_uniform_normal_ops() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let state = builder.parameter(0, u64::TY, &[2], "state")?;
    let (state, uniform) = state.sample_uniform(
        xla::RandomAlgorithm::ThreeFry,
        xla::ElementType::F32,
        &[10000],
        -2.,
        3.,
    )?;
    let (_, normal) =
        state.sample_normal(xla::RandomAlgorithm::ThreeFry, xla::ElementType::F32, &[10000])?;
    let exe = builder.tuple(&[&uniform, &normal])?.build()?.compile(&client)?;
    let seed = xla::Literal::vec1(&[42u64, 1337u64]);
    let r = &exe.execute::<xla::Literal>(&[seed])?[0];
    let uniform = r[0].to_literal_sync()?.to_vec::<f32>()?;
    let normal = r[1].to_literal_sync()?.to_vec::<f32>()?;
    // Uniform samples respect the bounds and have roughly the right moments.
    assert!(uniform.iter().all(|&x| (-2.0..3.0).contains(&x)));
    let mean = uniform.iter().sum::<f32>() / uniform.len() as f32;
    assert!((mean - 0.5).abs() < 0.05, "uniform mean {mean}");
    // Normal samples have roughly zero mean and unit variance.
    let mean = normal.iter().sum::<f32>() / normal.len() as f32;
    let var = normal.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / normal.len() as f32;
    assert!(mean.abs() < 0.05, "normal mean {mean}");
    assert!((var - 1.0).abs() < 0.1, "normal var {var}");
    // The two draws use different states and are uncorrelated with each other.
    let (u_mean, n_mean) = (0.5f32, 0f32);
    let cov =
        uniform.iter().zip(normal.iter()).map(|(u, n)| (u - u_mean) * (n - n_mean)).sum::<f32>()
            / uniform.len() as f32;
    assert!(cov.abs() < 0.05, "covariance {cov}");
    Ok(())
}
