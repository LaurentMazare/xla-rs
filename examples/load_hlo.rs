// This example is a conversion of examples/jax_cpp/main.cc from the jax repo.
// HLO files can be generated via the following command line in the jax repo.
// python \
//     jax/tools/jax_to_ir.py \
//     --fn examples.jax_cpp.prog.fn \
//     --input_shapes '[("x", "f32[2,2]"), ("y", "f32[2,2]")]' \
//     --constants '{"z": 2.0}' \
//     --ir_format HLO \
//     --ir_human_dest /tmp/fn_hlo.txt  \
//     --ir_dest /tmp/fn_hlo.pb
use anyhow::Result;
extern crate xla;

const USE_TEXT_FORMAT: bool = false;

fn main() -> Result<()> {
    xla::set_tf_min_log_level(xla::TfLogLevel::Warning);
    let client = xla::PjRtClient::cpu()?;
    println!("{} {} {}", client.platform_name(), client.platform_version(), client.device_count());
    let proto = if USE_TEXT_FORMAT {
        xla::HloModuleProto::from_text_file("examples/fn_hlo.txt")?
    } else {
        xla::HloModuleProto::from_proto_file("examples/fn_hlo.pb", true)?
    };
    let comp = xla::XlaComputation::from_proto(&proto);
    let result = client.compile(&comp)?;
    let x = xla::Literal::vec1(&[1f32, 2f32, 3f32, 4f32]).reshape(&[2, 2])?;
    let y = xla::Literal::vec1(&[1f32, 1f32, 1f32, 1f32]).reshape(&[2, 2])?;
    let result = result.execute::<xla::Literal>(&[x, y])?[0][0].to_literal_sync()?;
    let result = &result.to_tuple1()?;
    let shape = result.shape()?;
    println!("Result: {:?} {:?}", shape, result.to_vec::<f32>(),);
    Ok(())
}
