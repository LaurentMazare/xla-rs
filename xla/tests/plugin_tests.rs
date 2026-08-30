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

#[test]
fn plugin_client_adds_two_vectors() -> xla::Result<()> {
    let Some((device_type, path)) = plugin_from_env() else {
        return Ok(());
    };
    let client = PjRtClient::plugin(&device_type, &path)?;
    // No scalar constants, no implicit broadcast: y = x + z.
    let builder = XlaBuilder::new("plugin-add");
    let x = builder.parameter(0, ElementType::F32, &[4], "x")?;
    let z = builder.parameter(1, ElementType::F32, &[4], "z")?;
    let exe = client.compile(&x.add_(&z)?.build()?)?;
    let x_buf = client.buffer_from_host_buffer(&[1f32, 2., 3., 4.], &[4], None)?;
    let z_buf = client.buffer_from_host_buffer(&[10f32, 20., 30., 40.], &[4], None)?;
    let out = exe.execute_b(&[&x_buf, &z_buf])?;
    let out = out[0][0].to_literal_sync()?.to_vec::<f32>()?;
    assert_eq!(out, vec![11f32, 22., 33., 44.]);
    Ok(())
}

/// Reproduces the depformer's embedding pattern on a bf16 table: a gather
/// (`take`) on clamped indices, followed by a select that zeroes the rows
/// whose index was negative. Each variant is compiled and run separately so a
/// device fault points at one pattern.
#[test]
fn plugin_gather_patterns() -> xla::Result<()> {
    let Some((device_type, path)) = plugin_from_env() else {
        return Ok(());
    };
    let client = PjRtClient::plugin(&device_type, &path)?;
    let (rows, d, n) = (8001i64, 128i64, 32i64);
    let table_host: Vec<f32> = (0..rows * d).map(|i| ((i % 251) as f32) * 0.01).collect();
    let table_bf16: Vec<u8> = table_host
        .iter()
        .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let variants: [(&str, Vec<i32>, bool); 4] = [
        ("plain-valid", (0..n as i32).map(|i| i * 200).collect(), false),
        ("plain-last-row", vec![rows as i32 - 1; n as usize], false),
        ("clamp-select-valid", (0..n as i32).map(|i| i * 200).collect(), true),
        ("clamp-select-neg", (0..n as i32).map(|i| if i % 2 == 0 { -1 } else { i * 200 }).collect(), true),
    ];
    for (name, ids_host, clamp_select) in variants.iter() {
        let builder = XlaBuilder::new(name);
        let ids = builder.parameter(0, ElementType::S32, &[n], "ids")?;
        let table = builder.parameter(1, ElementType::Bf16, &[rows, d], "table")?;
        let out = if *clamp_select {
            let zero_ids = builder.c0(0i32)?.broadcast(&[n])?;
            let clamped = ids.max(&zero_ids)?;
            let emb = table.take(&clamped, 0)?;
            let zeros = builder.zero(ElementType::Bf16)?.broadcast(&[n, d])?;
            ids.lt(&zero_ids)?.broadcast_in_dim(&[n, d], &[0])?.select(&zeros, &emb)?
        } else {
            table.take(&ids, 0)?
        };
        let out = out.convert(xla::PrimitiveType::F32)?;
        let exe = client.compile(&out.build()?)?;
        let ids_buf = client.buffer_from_host_buffer(ids_host, &[n as usize], None)?;
        let table_buf = client.buffer_from_host_raw_bytes(
            ElementType::Bf16,
            &table_bf16,
            &[rows as usize, d as usize],
            None,
        )?;
        let res = exe.execute_b(&[&ids_buf, &table_buf]);
        match res {
            Ok(out) => {
                let v = out[0][0].to_literal_sync()?.to_vec::<f32>()?;
                // Check row 1 against the host table (or zero for a negative id).
                let id = ids_host[1];
                let expect = if id < 0 { 0.0 } else { table_host[(id as i64 * d) as usize] };
                let got = v[d as usize];
                eprintln!("{name}: ok, row1[0]={got} (expect ~{expect})");
            }
            Err(e) => eprintln!("{name}: FAILED {e}"),
        }
    }
    Ok(())
}

/// The depformer's sampling chain, piece by piece: argmax against a host
/// reference, Gumbel sampling (rng_bit_generator DEFAULT) token range, and
/// the sampled tokens feeding the embedding gather.
#[test]
fn plugin_sampling_chain() -> xla::Result<()> {
    let Some((device_type, path)) = plugin_from_env() else {
        return Ok(());
    };
    let client = PjRtClient::plugin(&device_type, &path)?;
    let (b, vocab, d) = (16i64, 2048i64, 128i64);
    // Deterministic pseudo-random logits with a clear per-row maximum.
    let mut logits_host = vec![0f32; (b * vocab) as usize];
    let mut x: u32 = 12345;
    for v in logits_host.iter_mut() {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((x >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0;
    }
    let host_argmax: Vec<i32> = (0..b)
        .map(|r| {
            let row = &logits_host[(r * vocab) as usize..((r + 1) * vocab) as usize];
            row.iter().enumerate().fold((0usize, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc }).0 as i32
        })
        .collect();
    let logits_buf = client.buffer_from_host_buffer(&logits_host, &[b as usize, vocab as usize], None)?;

    // A: plain argmax.
    {
        let builder = XlaBuilder::new("argmax");
        let logits = builder.parameter(0, ElementType::F32, &[b, vocab], "logits")?;
        let tok = logits.argmax(ElementType::S32, -1)?.reshape(&[b])?;
        let exe = client.compile(&tok.build()?)?;
        match exe.execute_b(&[&logits_buf]) {
            Ok(out) => {
                let got = out[0][0].to_literal_sync()?.to_vec::<i32>()?;
                eprintln!("argmax: ok, matches host = {}, got[..4]={:?} host[..4]={:?}", got == host_argmax, &got[..4], &host_argmax[..4]);
            }
            Err(e) => eprintln!("argmax: FAILED {e}"),
        }
    }
    // B: gumbel sampling with the DEFAULT rng.
    let rng_host = [0x1234_5678_9abc_def0u64, 0x0fed_cba9_8765_4321u64];
    let rng_buf = client.buffer_from_host_buffer(&rng_host, &[2], None)?;
    let temp_host = vec![0.6f32; b as usize];
    let temp_buf = client.buffer_from_host_buffer(&temp_host, &[b as usize], None)?;
    {
        let builder = XlaBuilder::new("gumbel");
        let logits = builder.parameter(0, ElementType::F32, &[b, vocab], "logits")?;
        let temp = builder.parameter(1, ElementType::F32, &[b], "temp")?;
        let rng = builder.parameter(2, ElementType::U64, &[2], "rng")?;
        let (new_rng, u) = rng.sample_uniform(xla::RandomAlgorithm::Default, ElementType::F32, &[b, vocab], 1e-7, 0.999)?;
        let noise = u.log()?.neg()?.log()?;
        let t = temp.broadcast_in_dim(&[b, vocab], &[0])?;
        let shifted = logits.sub_(&noise.mul_(&t)?)?;
        let tok = shifted.argmax(ElementType::S32, -1)?.reshape(&[b])?;
        let exe = client.compile(&builder.tuple(&[&tok, &u, &new_rng])?.build()?)?;
        match exe.execute_b(&[&logits_buf, &temp_buf, &rng_buf]) {
            Ok(out) => {
                let tok = out[0][0].to_literal_sync()?.to_vec::<i32>()?;
                let u = out[0][1].to_literal_sync()?.to_vec::<f32>()?;
                let umin = u.iter().cloned().fold(f32::MAX, f32::min);
                let umax = u.iter().cloned().fold(f32::MIN, f32::max);
                let nan = u.iter().filter(|v| v.is_nan()).count();
                let bad: Vec<i32> = tok.iter().cloned().filter(|&t| t < 0 || t >= vocab as i32).collect();
                eprintln!("gumbel: ok, tokens={:?} out-of-range={:?} u in [{umin}, {umax}] nan={nan}", &tok[..8], bad);
            }
            Err(e) => eprintln!("gumbel: FAILED {e}"),
        }
    }
    // C: sampled tokens (dup to 2b) into the embedding gather.
    {
        let table_host: Vec<u8> = (0..(vocab + 1) * d).flat_map(|i| ((((i % 251) as f32 * 0.01).to_bits() >> 16) as u16).to_le_bytes()).collect();
        let table_buf = client.buffer_from_host_raw_bytes(ElementType::Bf16, &table_host, &[(vocab + 1) as usize, d as usize], None)?;
        let builder = XlaBuilder::new("chain");
        let logits = builder.parameter(0, ElementType::F32, &[b, vocab], "logits")?;
        let temp = builder.parameter(1, ElementType::F32, &[b], "temp")?;
        let rng = builder.parameter(2, ElementType::U64, &[2], "rng")?;
        let table = builder.parameter(3, ElementType::Bf16, &[vocab + 1, d], "table")?;
        let (_new_rng, u) = rng.sample_uniform(xla::RandomAlgorithm::Default, ElementType::F32, &[b, vocab], 1e-7, 0.999)?;
        let noise = u.log()?.neg()?.log()?;
        let t = temp.broadcast_in_dim(&[b, vocab], &[0])?;
        let tok = logits.sub_(&noise.mul_(&t)?)?.argmax(ElementType::S32, -1)?.reshape(&[b])?;
        let ids = tok.concat_in_dim(&[&tok], 0)?; // [2b]
        let n = 2 * b;
        let zero_ids = builder.c0(0i32)?.broadcast(&[n])?;
        let emb = table.take(&ids.max(&zero_ids)?, 0)?;
        let zeros = builder.zero(ElementType::Bf16)?.broadcast(&[n, d])?;
        let emb = ids.lt(&zero_ids)?.broadcast_in_dim(&[n, d], &[0])?.select(&zeros, &emb)?;
        let out = emb.convert(xla::PrimitiveType::F32)?.reduce_sum(&[1], false)?;
        let exe = client.compile(&builder.tuple(&[&out, &tok])?.build()?)?;
        match exe.execute_b(&[&logits_buf, &temp_buf, &rng_buf, &table_buf]) {
            Ok(out) => {
                let s = out[0][0].to_literal_sync()?.to_vec::<f32>()?;
                let tok = out[0][1].to_literal_sync()?.to_vec::<i32>()?;
                eprintln!("chain: ok, tokens[..4]={:?} rowsum[..4]={:?}", &tok[..4], &s[..4]);
            }
            Err(e) => eprintln!("chain: FAILED {e}"),
        }
    }
    Ok(())
}

/// The sampling chain with the rng state aliased in-place (input/output
/// alias), as the depformer step does.
#[test]
fn plugin_sampling_chain_aliased() -> xla::Result<()> {
    let Some((device_type, path)) = plugin_from_env() else {
        return Ok(());
    };
    let client = PjRtClient::plugin(&device_type, &path)?;
    let (b, vocab, d) = (16i64, 2048i64, 128i64);
    let logits_host = vec![0.5f32; (b * vocab) as usize];
    let logits_buf = client.buffer_from_host_buffer(&logits_host, &[b as usize, vocab as usize], None)?;
    let temp_buf = client.buffer_from_host_buffer(&vec![0.6f32; b as usize], &[b as usize], None)?;
    let table_host: Vec<u8> = (0..(vocab + 1) * d).flat_map(|i| ((((i % 251) as f32 * 0.01).to_bits() >> 16) as u16).to_le_bytes()).collect();
    let table_buf = client.buffer_from_host_raw_bytes(ElementType::Bf16, &table_host, &[(vocab + 1) as usize, d as usize], None)?;
    let builder = XlaBuilder::new("chain-aliased");
    let logits = builder.parameter(0, ElementType::F32, &[b, vocab], "logits")?;
    let temp = builder.parameter(1, ElementType::F32, &[b], "temp")?;
    let table = builder.parameter(2, ElementType::Bf16, &[vocab + 1, d], "table")?;
    let rng = builder.parameter(3, ElementType::U64, &[2], "rng")?;
    let (new_rng, u) = rng.sample_uniform(xla::RandomAlgorithm::Default, ElementType::F32, &[b, vocab], 1e-7, 0.999)?;
    let noise = u.log()?.neg()?.log()?;
    let t = temp.broadcast_in_dim(&[b, vocab], &[0])?;
    let tok = logits.sub_(&noise.mul_(&t)?)?.argmax(ElementType::S32, -1)?.reshape(&[b])?;
    let ids = tok.concat_in_dim(&[&tok], 0)?;
    let n = 2 * b;
    let zero_ids = builder.c0(0i32)?.broadcast(&[n])?;
    let emb = table.take(&ids.max(&zero_ids)?, 0)?;
    let zeros = builder.zero(ElementType::Bf16)?.broadcast(&[n, d])?;
    let emb = ids.lt(&zero_ids)?.broadcast_in_dim(&[n, d], &[0])?.select(&zeros, &emb)?;
    let codes = tok.convert(xla::PrimitiveType::S64)?.reshape(&[b, 1, 1])?;
    let out = emb.convert(xla::PrimitiveType::F32)?.reduce_sum(&[1], false)?;
    // Output 0: codes, output 1: new rng (aliased to param 3), output 2: embedding sums.
    builder.setup_alias(1, 3);
    let exe = client.compile(&builder.tuple(&[&codes, &new_rng, &out])?.build()?)?;
    let mut rng_buf = client.buffer_from_host_buffer(&[0x1234_5678_9abc_def0u64, 0x0fed_cba9_8765_4321u64], &[2], None)?;
    for step in 0..3 {
        match exe.execute_b(&[&logits_buf, &temp_buf, &table_buf, &rng_buf]) {
            Ok(mut out) => {
                let mut outs = out.remove(0);
                let codes = outs[0].to_literal_sync()?.to_vec::<i64>()?;
                let sums = outs[2].to_literal_sync()?.to_vec::<f32>()?;
                eprintln!("aliased step {step}: ok codes[..4]={:?} sums[..2]={:?}", &codes[..4], &sums[..2]);
                rng_buf = outs.remove(1);
            }
            Err(e) => {
                eprintln!("aliased step {step}: FAILED {e}");
                break;
            }
        }
    }
    Ok(())
}
