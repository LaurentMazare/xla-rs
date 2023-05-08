use anyhow::Result;
extern crate xla;
use xla::ArrayElement;

fn main() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    loop {
        let builder = xla::XlaBuilder::new("test");
        let x = builder.parameter(0, f32::TY, &[2], "x")?;
        let sum = x.reduce_sum(&[], false)?.build()?.compile(&client)?;
        let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
        let result = sum.execute::<xla::Literal>(&[input])?;
        println!("1");
        let result = result[0][0].to_literal_sync()?;
        drop(sum);
        assert_eq!(result.to_vec::<f32>()?, [4.2, 1.337]);

        let builder = xla::XlaBuilder::new("test");
        let x = builder.parameter(0, f32::TY, &[-2], "x")?;
        let sum = x.reduce_sum(&[0], false)?.build()?.compile(&client)?;
        let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
        let result = sum.execute::<xla::Literal>(&[input])?;
        println!("2");
        let result = result[0][0].to_literal_sync()?;
        drop(sum);
        assert_eq!(result.to_vec::<f32>()?, [5.5369997]);
        // Dimensions got reduced.
        assert_eq!(result.array_shape()?.dims(), []);

        let builder = xla::XlaBuilder::new("test");
        let x = builder.parameter(0, f32::TY, &[-2], "x")?;
        let sum = x.reduce_sum(&[0], true)?.build()?.compile(&client)?;
        let input = xla::Literal::vec1(&[4.2f32, 1.337f32]);
        let result = sum.execute::<xla::Literal>(&[input])?;
        println!("3");
        let result = result[0][0].to_literal_sync()?;
        drop(sum);
        assert_eq!(result.to_vec::<f32>()?, [5.5369997]);
        // keep_dims = true in this case.
        assert_eq!(result.array_shape()?.dims(), [1]);
        println!("Done!");
    }
}
