use anyhow::Result;
extern crate xla;

fn global_data_f32(f: f32) -> Result<xla::GlobalData> {
    xla::Literal::from(f).transfer_to_server()
}

fn main() -> Result<()> {
    let xla_builder = xla::XlaBuilder::new("test");
    let cst42 = xla::XlaBuilder::constant_r0(&xla_builder, 42.);
    let cst43 = xla::XlaBuilder::constant_r1(&xla_builder, 2, 43.);
    let sum = cst42.add(&cst43);
    println!("Shape: {:?}", xla_builder.get_shape(&sum));
    let computation = xla_builder.build(&sum)?;
    let result = computation.run(&[])?;
    println!("Result: {:?}", result.get_first_element_f32());
    let param = xla_builder.parameter(0, &xla::Shape::new::<f32>(vec![]), "p");
    let sum = param.add(&param);
    let sum = sum.sqrt();
    let computation = xla_builder.build(&sum)?;
    let result = computation.run(&[global_data_f32(12.0)?])?;
    println!("Result: {:?}", result.get_first_element_f32());
    let result = computation.run(&[global_data_f32(13.0)?])?;
    println!("Result: {:?}", result.get_first_element_f32());
    Ok(())
}
