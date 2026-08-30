//! Exercises a PJRT C-API plugin named by the environment:
//! `XLA_PJRT_PLUGIN=<device_type>:<path/to/plugin.so>` (skipped when unset).
use xla::{ElementType, PjRtClient, XlaBuilder};

fn plugin_from_env() -> Option<(String, String)> {
    let spec = std::env::var("XLA_PJRT_PLUGIN").ok()?;
    let (device_type, path) = spec.split_once(':')?;
    Some((device_type.to_string(), path.to_string()))
}

#[test]
fn plugin_client_runs_a_computation() -> xla::Result<()> {
    let Some((device_type, path)) = plugin_from_env() else {
        eprintln!("XLA_PJRT_PLUGIN is not set, skipping");
        return Ok(());
    };
    let client = PjRtClient::plugin(&device_type, &path)?;
    eprintln!(
        "platform {} ({}), {} device(s)",
        client.platform_name(),
        client.platform_version(),
        client.addressable_device_count()
    );
    assert!(client.addressable_device_count() > 0);
    // y = 2 * x + 1 over an f32 vector, executed from a host buffer.
    let builder = XlaBuilder::new("plugin-test");
    let x = builder.parameter(0, ElementType::F32, &[4], "x")?;
    let y = x.mul_(&builder.c0(2f32)?)?.add_(&builder.c0(1f32)?)?;
    let exe = client.compile(&y.build()?)?;
    let x_buf = client.buffer_from_host_buffer(&[1f32, 2., 3., 4.], &[4], None)?;
    let out = exe.execute_b(&[&x_buf])?;
    let out = out[0][0].to_literal_sync()?.to_vec::<f32>()?;
    assert_eq!(out, vec![3f32, 5., 7., 9.]);
    // A second client on the same, already registered, plugin.
    let again = PjRtClient::plugin(&device_type, &path)?;
    assert_eq!(again.platform_name(), client.platform_name());
    Ok(())
}
