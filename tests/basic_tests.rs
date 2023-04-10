use xla::{ElementType, Result};

#[test]
fn add_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let cst42 = builder.constant_r0(42f32);
    let cst43 = builder.constant_r1c(43f32, 2);
    let sum = cst42.add(&cst43);
    let computation = sum.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<xla::PjRtBuffer>(&[])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.element_count(), 2);
    assert_eq!(result.shape()?, xla::Shape::new::<f32>(vec![2]));
    assert_eq!(result.get_first_element::<f32>()?, 85.);
    assert_eq!(result.to_vec::<f32>()?, [85., 85.]);
    Ok(())
}

#[test]
fn sum_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::PRIMITIVE_TYPE, &[2], "x");
    let sum = x.sum(&[])?.build()?.compile(&client)?;
    let input = xla::Literal::vec(&[4.2f32, 1.337f32]);
    let result = sum.execute_literal::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [4.2, 1.337]);

    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::PRIMITIVE_TYPE, &[2], "x");
    let sum = x.sum(&[0])?.build()?.compile(&client)?;
    let input = xla::Literal::vec(&[4.2f32, 1.337f32]);
    let result = sum.execute_literal::<xla::Literal>(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.to_vec::<f32>()?, [5.5369997]);
    Ok(())
}
