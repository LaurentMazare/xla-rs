//! Runs a Neuron NKI kernel through the PJRT plugin as an
//! `AwsNeuronCustomNativeKernel` custom call. Requires two env vars (skipped
//! otherwise):
//!   XLA_PJRT_PLUGIN=<device_type>:<path/to/libneuronpjrt.so>
//!   NKI_RING_CFG=<path/to/backend_config>  (from FrameworkKernel::dump_config)
//!
//! The kernel is the TTS ring-cache write: kv [32, 384, 128] bf16 written into
//! cache slot `NKI_RING_SLOT` (default 137) of out [32, 384, 200, 128] bf16.
//! Inputs and the slot slice go through f32 converts so the host side stays in
//! f32; the rest of `out` is uninitialized and never read.
use xla::{ElementType, PjRtClient, XlaBuilder};

const G: i64 = 32;
const BH: i64 = 384;
const C: i64 = 200;
const HD: i64 = 128;

#[test]
fn nki_ring_write_via_custom_call() -> xla::Result<()> {
    let Some(spec) = std::env::var("XLA_PJRT_PLUGIN").ok() else {
        eprintln!("XLA_PJRT_PLUGIN is not set, skipping");
        return Ok(());
    };
    let Some(cfg_path) = std::env::var("NKI_RING_CFG").ok() else {
        eprintln!("NKI_RING_CFG is not set, skipping");
        return Ok(());
    };
    let (device_type, path) = spec.split_once(':').expect("bad XLA_PJRT_PLUGIN");
    let slot: i64 = std::env::var("NKI_RING_SLOT").ok().and_then(|s| s.parse().ok()).unwrap_or(137);
    let opaque = std::fs::read(&cfg_path).expect("cannot read NKI_RING_CFG");
    let client = PjRtClient::plugin(device_type, path)?;

    let builder = XlaBuilder::new("nki-ring-write");
    let kvf = builder.parameter(0, ElementType::F32, &[G, BH, HD], "kvf")?;
    let kv = kvf.convert(xla::PrimitiveType::Bf16)?;
    let out = builder.custom_call(
        "AwsNeuronCustomNativeKernel",
        &[kv],
        ElementType::Bf16,
        &[G, BH, C, HD],
        &opaque,
    )?;
    // Only the written slot is defined; slice it out and widen for the host.
    let written = out
        .slice_in_dim(slot, slot + 1, 1, 2)?
        .reshape(&[G, BH, HD])?
        .convert(xla::PrimitiveType::F32)?;
    let exe = client.compile(&written.build()?)?;

    let n = (G * BH * HD) as usize;
    // bf16-exact values so the round-trip compares clean.
    let kv_host: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) - 125.0).collect();
    let kv_buf =
        client.buffer_from_host_buffer(&kv_host, &[G as usize, BH as usize, HD as usize], None)?;
    let out = exe.execute_b(&[&kv_buf])?;
    let got = out[0][0].to_literal_sync()?.to_vec::<f32>()?;
    assert_eq!(got.len(), n);
    let bad = got.iter().zip(kv_host.iter()).filter(|(a, b)| a != b).count();
    eprintln!("mismatched elements: {bad}/{n}");
    assert_eq!(bad, 0, "kernel output does not match the written rows");

    // Steady-state timing of the whole executable (upload + kernel + slice).
    let iters = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let out = exe.execute_b(&[&kv_buf])?;
        out[0][0].to_literal_sync()?;
    }
    let us = t0.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("custom-call ring write: {us:.0} us/exec over {iters} iters");
    Ok(())
}

#[test]
fn nki_ring_write_dynamic_slot() -> xla::Result<()> {
    // Same write, but the slot is a runtime operand: one executable serves
    // every ring position. Config from the dynamic-slot kernel
    // (out is row-flattened to [G*BH*C, HD]).
    let Some(spec) = std::env::var("XLA_PJRT_PLUGIN").ok() else {
        eprintln!("XLA_PJRT_PLUGIN is not set, skipping");
        return Ok(());
    };
    let Some(cfg_path) = std::env::var("NKI_RING_DYN_CFG").ok() else {
        eprintln!("NKI_RING_DYN_CFG is not set, skipping");
        return Ok(());
    };
    let (device_type, path) = spec.split_once(':').expect("bad XLA_PJRT_PLUGIN");
    let opaque = std::fs::read(&cfg_path).expect("cannot read NKI_RING_DYN_CFG");
    let client = PjRtClient::plugin(device_type, path)?;

    let builder = XlaBuilder::new("nki-ring-write-dyn");
    let kvf = builder.parameter(0, ElementType::F32, &[G, BH, HD], "kvf")?;
    let slot = builder.parameter(1, ElementType::S32, &[1, 1], "slot")?;
    let kv = kvf.convert(xla::PrimitiveType::Bf16)?;
    let out = builder.custom_call(
        "AwsNeuronCustomNativeKernel",
        &[kv, slot.clone()],
        ElementType::Bf16,
        &[G * BH * C, HD],
        &opaque,
    )?;
    // Read back the written slot with a dynamic slice at the same runtime index.
    let out4 = out.reshape(&[G, BH, C, HD])?;
    let zero = builder.c0(0i32)?;
    let slot_scalar = slot.reshape(&[])?;
    let written = out4
        .dynamic_slice(&[zero.clone(), zero.clone(), slot_scalar, zero], &[G, BH, 1, HD])?
        .reshape(&[G, BH, HD])?
        .convert(xla::PrimitiveType::F32)?;
    let exe = client.compile(&written.build()?)?;

    let n = (G * BH * HD) as usize;
    let kv_host: Vec<f32> = (0..n).map(|i| ((i % 251) as f32) - 125.0).collect();
    let kv_buf =
        client.buffer_from_host_buffer(&kv_host, &[G as usize, BH as usize, HD as usize], None)?;
    for slot_val in [137i32, 42, 0, 199] {
        let slot_buf = client.buffer_from_host_buffer(&[slot_val], &[1, 1], None)?;
        let out = exe.execute_b(&[&kv_buf, &slot_buf])?;
        let got = out[0][0].to_literal_sync()?.to_vec::<f32>()?;
        let bad = got.iter().zip(kv_host.iter()).filter(|(a, b)| a != b).count();
        eprintln!("slot {slot_val}: mismatched {bad}/{n}");
        assert_eq!(bad, 0, "dynamic-slot write failed at slot {slot_val}");
    }

    let iters = 50;
    let slot_buf = client.buffer_from_host_buffer(&[137i32], &[1, 1], None)?;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let out = exe.execute_b(&[&kv_buf, &slot_buf])?;
        out[0][0].to_literal_sync()?;
    }
    let us = t0.elapsed().as_micros() as f64 / iters as f64;
    eprintln!("dynamic-slot ring write: {us:.0} us/exec over {iters} iters");
    Ok(())
}
