use xla::{ArrayElement, Result};

#[test]
fn while_op() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    let builder = xla::XlaBuilder::new("test");
    let cond = {
        let builder = xla::XlaBuilder::new("cond");
        let x = builder.parameter(0, i32::TY, &[], "x")?;
        x.le(&builder.constant_r0(10i32)?)?.build()?
    };
    let body = {
        let builder = xla::XlaBuilder::new("cond");
        let x = builder.parameter(0, i32::TY, &[], "x")?;
        (x + builder.constant_r0(1i32)?)?.build()?
    };
    let init = builder.constant_r0(0i32)?;
    let w = xla::XlaOp::while_(cond, body, init)?;
    let computation = w.build()?;
    let result = client.compile(&computation)?;
    let result = result.execute::<xla::Literal>(&[])?;
    let result = result[0][0].to_literal_sync()?;
    assert_eq!(result.element_count(), 1);
    assert_eq!(result.shape()?, xla::Shape::array::<i32>(vec![]));
    assert_eq!(result.to_vec::<i32>()?, [11]);
    Ok(())
}
