use anyhow::Result;
extern crate xla;

fn main() -> Result<()> {
    xla::set_tf_min_log_level(xla::TfLogLevel::Warning);

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
    let cst42 = xla_builder.constant_r0(42f32)?;
    let cst43 = xla_builder.constant_r1(&[43f32, 44f32])?;
    let sum = (cst42 + cst43)?;
    println!("Shape: {:?}", xla_builder.get_shape(&sum));
    let sum = sum.build()?;
    let result = client.compile(&sum)?;
    let result = &result.execute::<xla::Literal>(&[])?[0][0].to_literal_sync()?;
    let shape = result.shape()?;
    println!(
        "Result: {:?} {:?} {}",
        shape,
        result.to_vec::<f32>(),
        result.get_first_element::<f32>()?,
    );
    let param = xla_builder.parameter_s(0, &xla::Shape::array::<f32>(vec![]), "p")?;
    let sum = param.add_(&param)?;
    let sum = sum.sqrt()?.build()?;
    let result = client.compile(&sum)?;
    let result = result.execute(&[xla::Literal::from(12f32)])?[0][0].to_literal_sync()?;
    println!("Result: {:?} {:?}", result.shape(), result.get_first_element::<f32>());
    let result = client.compile(&sum)?;
    let result = result.execute(&[xla::Literal::from(13f32)])?[0][0].to_literal_sync()?;
    println!("Result: {:?} {:?}", result.shape(), result.get_first_element::<f32>());
    Ok(())
}
