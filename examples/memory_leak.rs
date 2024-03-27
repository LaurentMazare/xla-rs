use anyhow::Result;
extern crate xla;
use xla::ArrayElement;

fn main() -> Result<()> {
    let client = xla::PjRtClient::gpu(0.8, true)?;
    let builder = xla::XlaBuilder::new("test");
    let x = builder.parameter(0, f32::TY, &[1000000], "x")?;
    let comp = x.build()?;
    let exe = comp.compile(&client)?;
    let mut x_val = xla::Literal::vec1(&[0f32; 1000000]);
    let mut count = 0;
    loop {
        let xla_buffers = exe.execute(&[&x_val])?;
        let literal_out = xla_buffers[0][0].to_literal_sync()?;
        x_val = literal_out;
        println!("{}", count);
        count += 1;
        //Ok(());
    }
}