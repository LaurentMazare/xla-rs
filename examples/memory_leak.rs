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
    let mut x_buf = client.buffer_from_host_literal(Some(&client.devices()[0]), &x_val)?;
    let mut count = 0;
    loop {
        let mut xla_buffers = exe.execute_b(&[&x_buf])?;
        x_buf = xla_buffers.pop().unwrap().pop().unwrap();
        println!("{}", count);
        count += 1;
        //Ok(());
    }
}