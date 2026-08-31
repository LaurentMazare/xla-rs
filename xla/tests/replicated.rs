//! Replicated (SPMD) compilation and execution, gated on `XLA_PJRT_PLUGIN`
//! pointing at a PJRT plugin exposing at least two addressable devices (e.g.
//! `neuron:/path/to/libneuronpjrt.so`). Skipped otherwise.
use xla::{ElementType, PjRtClient, XlaBuilder, XlaComputation};

fn plugin_client() -> Option<PjRtClient> {
    let spec = std::env::var("XLA_PJRT_PLUGIN").ok()?;
    let (device_type, path) = spec.split_once(':')?;
    let client = PjRtClient::plugin(device_type, path).ok()?;
    if client.addressable_device_count() < 2 {
        eprintln!("fewer than 2 devices, skipping");
        return None;
    }
    Some(client)
}

fn add_computation(ty: ElementType) -> xla::Result<XlaComputation> {
    let b = XlaBuilder::new("add");
    let x = b.parameter(0, ty, &[], "x")?;
    let y = b.parameter(1, ty, &[], "y")?;
    x.add_(&y)?.build()
}

/// Two replicas each double their own input, then all-reduce the results:
/// both replicas must observe the same cross-replica sum.
#[test]
fn all_reduce_smoke() -> xla::Result<()> {
    let Some(client) = plugin_client() else { return Ok(()) };
    let builder = XlaBuilder::new("replicated-smoke");
    let x = builder.parameter(0, ElementType::F32, &[4], "x")?;
    let y = x.mul_(&builder.c0(2f32)?)?.all_reduce(&add_computation(ElementType::F32)?)?;
    let exe = client.compile_replicated(&y.build()?, 2)?;
    let devs = client.addressable_devices();
    let x0 = client.buffer_from_host_buffer(&[1f32, 2., 3., 4.], &[4], Some(&devs[0]))?;
    let x1 = client.buffer_from_host_buffer(&[10f32, 20., 30., 40.], &[4], Some(&devs[1]))?;
    let out = exe.execute_replicated_b(&[vec![&x0], vec![&x1]])?;
    let r0 = out[0][0].to_literal_sync()?.to_vec::<f32>()?;
    let r1 = out[1][0].to_literal_sync()?.to_vec::<f32>()?;
    assert_eq!(r0, vec![22f32, 44., 66., 88.]);
    assert_eq!(r1, r0);
    Ok(())
}
