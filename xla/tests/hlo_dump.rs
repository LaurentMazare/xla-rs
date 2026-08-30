//! Dumps an HLO module proto to text: `HLO_PB=<in.pb> HLO_OUT=<out.txt>`.
#[test]
fn dump_hlo_text() -> xla::Result<()> {
    let (Ok(pb), Ok(out)) = (std::env::var("HLO_PB"), std::env::var("HLO_OUT")) else {
        return Ok(());
    };
    let module = xla::HloModuleProto::from_proto_file(&pb, true)?;
    module.to_text_file(&out)?;
    Ok(())
}
