use anyhow::Result;
extern crate xla;

fn main() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    println!("{} {} {}", client.platform_name(), client.platform_version(), client.device_count());
    for device in client.devices().iter() {
        println!(
            "{} {} {} {}",
            device.id(),
            device.to_string(),
            device.debug_string(),
            device.kind()
        )
    }
    let xla_builder = xla::XlaBuilder::new("test");
    let cst42 = xla::XlaBuilder::constant_r0(&xla_builder, 42.);
    let cst43 = xla::XlaBuilder::constant_r1(&xla_builder, &[43., 44.]);
    let sum = cst42.add(&cst43);
    println!("Shape: {:?}", xla_builder.get_shape(&sum));
    let computation = xla_builder.build(&sum)?;
    let result = client.compile(&computation)?;
    let result = &result.execute::<xla::PjRtBuffer>(&[])?[0][0].to_literal_sync()?;
    let shape = result.shape()?;
    println!("Result: {:?} {:?} {}", shape, result.to_vec::<f32>(), result.get_first_element_f32());
    let param = xla_builder.parameter(0, &xla::Shape::new::<f32>(vec![]), "p");
    let sum = param.add(&param);
    let sum = sum.sqrt();
    let computation = xla_builder.build(&sum)?;
    let result = client.compile(&computation)?;
    let result = result.execute_literal(&[xla::Literal::from(12.0)])?[0][0].to_literal_sync()?;
    println!("Result: {:?} {:?}", result.shape(), result.get_first_element_f32());
    let result = client.compile(&computation)?;
    let result = result.execute_literal(&[xla::Literal::from(13.0)])?[0][0].to_literal_sync()?;
    println!("Result: {:?} {:?}", result.shape(), result.get_first_element_f32());
    Ok(())
}
