#[test]
fn assign_ops() {
    let client = xla::PjRtClient::cpu().unwrap();
    let xla_builder = xla::XlaBuilder::new("test");
    let cst42 = xla::XlaBuilder::constant_r0(&xla_builder, 42f32);
    let cst43 = xla::XlaBuilder::constant_r1c(&xla_builder, 43f32, 2);
    let sum = cst42.add(&cst43);
    let computation = xla_builder.build(&sum).unwrap();
    let result = client.compile(&computation).unwrap();
    let result = result.execute::<xla::PjRtBuffer>(&[]).unwrap();
    let result = result[0][0].to_literal_sync().unwrap();
    assert_eq!(result.element_count(), 2);
    assert_eq!(result.shape().unwrap(), xla::Shape::new::<f32>(vec![2]));
    assert_eq!(result.get_first_element::<f32>(), 85.)
}
