#[test]
fn assign_ops() {
    let xla_builder = xla::XlaBuilder::new("test");
    let cst42 = xla::XlaBuilder::constant_r0(&xla_builder, 42.);
    let cst43 = xla::XlaBuilder::constant_r1(&xla_builder, 2, 43.);
    let sum = cst42.add(&cst43);
    let computation = xla_builder.build(&sum).unwrap();
    let result = computation.run(&[]).unwrap();
    assert_eq!(result.element_count(), 2);
    assert_eq!(result.shape().unwrap(), xla::Shape::new::<f32>(vec![2]));
    assert_eq!(result.get_first_element_f32(), 85.)
}
