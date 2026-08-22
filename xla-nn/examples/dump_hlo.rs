// The counterpart of load_hlo.rs: build a computation with the XlaBuilder and
// export it as HLO or StableHLO. The generated files can be inspected or fed
// back in via `HloModuleProto::from_text_file` / `from_proto_file`.
use anyhow::Result;
extern crate xla;

use xla::ArrayElement;

fn main() -> Result<()> {
    xla::set_tf_min_log_level(xla::TfLogLevel::Warning);
    let builder = xla::XlaBuilder::new("fn");
    let x = builder.parameter(0, f32::TY, &[2, 2], "x")?;
    let y = builder.parameter(1, f32::TY, &[2, 2], "y")?;
    let z = builder.constant_r0(2f32)?;
    let proto = ((&x + &y)? * z)?.build()?.proto();

    println!("=== HLO ===\n{}", proto.to_string()?);
    println!("=== StableHLO ===\n{}", proto.to_stablehlo_string()?);

    proto.to_text_file("/tmp/fn_hlo.txt")?;
    proto.to_proto_file("/tmp/fn_hlo.pb", true)?;
    println!("written to /tmp/fn_hlo.txt and /tmp/fn_hlo.pb");
    Ok(())
}
