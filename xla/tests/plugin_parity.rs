//! Numerical parity of a PJRT plugin against the in-process CPU client, on
//! the op patterns the moshi/mimi graphs are made of. Gated on
//! `XLA_PJRT_PLUGIN=<device_type>:<path>`; prints one line per case.
use xla::{ElementType, PjRtBuffer, PjRtClient, PrimitiveType, Result, XlaBuilder, XlaOp};

fn plugin_from_env() -> Option<(String, String)> {
    let spec = std::env::var("XLA_PJRT_PLUGIN").ok()?;
    let (d, p) = spec.split_once(':')?;
    Some((d.to_string(), p.to_string()))
}

/// Deterministic pseudo-random floats in `[-scale, scale]`.
fn rand(seed: u32, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| (((f.to_bits() + 0x8000) >> 16) as u16).to_le_bytes()).collect()
}

enum In {
    F32(Vec<i64>, Vec<f32>),
    Bf16(Vec<i64>, Vec<f32>),
    S32(Vec<i64>, Vec<i32>),
}

impl In {
    fn ty(&self) -> ElementType {
        match self {
            In::F32(..) => ElementType::F32,
            In::Bf16(..) => ElementType::Bf16,
            In::S32(..) => ElementType::S32,
        }
    }
    fn dims(&self) -> &[i64] {
        match self {
            In::F32(d, _) | In::Bf16(d, _) => d,
            In::S32(d, _) => d,
        }
    }
    fn upload(&self, client: &PjRtClient) -> Result<PjRtBuffer> {
        let dims: Vec<usize> = self.dims().iter().map(|d| *d as usize).collect();
        match self {
            In::F32(_, v) => client.buffer_from_host_buffer(v, &dims, None),
            In::Bf16(_, v) => {
                client.buffer_from_host_raw_bytes(ElementType::Bf16, &bf16_bytes(v), &dims, None)
            }
            In::S32(_, v) => client.buffer_from_host_buffer(v, &dims, None),
        }
    }
}

type Build = Box<dyn Fn(&XlaBuilder, &[XlaOp]) -> Result<XlaOp>>;

fn run(client: &PjRtClient, name: &str, inputs: &[In], build: &Build) -> Result<Vec<f32>> {
    let builder = XlaBuilder::new(name);
    let params: Vec<XlaOp> = inputs
        .iter()
        .enumerate()
        .map(|(i, inp)| builder.parameter(i as i64, inp.ty(), inp.dims(), &format!("p{i}")))
        .collect::<Result<_>>()?;
    let out = build(&builder, &params)?.convert(PrimitiveType::F32)?;
    let exe = client.compile(&out.build()?)?;
    let bufs: Vec<PjRtBuffer> = inputs.iter().map(|i| i.upload(client)).collect::<Result<_>>()?;
    let refs: Vec<&PjRtBuffer> = bufs.iter().collect();
    let out = exe.execute_b(&refs)?;
    out[0][0].to_literal_sync()?.to_vec::<f32>()
}

fn compare(name: &str, a: &[f32], b: &[f32]) {
    let nan_b = b.iter().filter(|v| v.is_nan()).count();
    let nan_a = a.iter().filter(|v| v.is_nan()).count();
    let scale = a.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let (mut max_abs, mut argmax) = (0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max_abs || (d.is_nan() && !max_abs.is_nan()) {
            max_abs = d;
            argmax = i;
        }
    }
    let rel = if scale > 0.0 { max_abs / scale } else { max_abs };
    let verdict = if nan_b > nan_a || rel > 2e-2 || rel.is_nan() {
        "MISMATCH"
    } else if rel > 2e-3 {
        "loose"
    } else {
        "ok"
    };
    eprintln!(
        "{verdict:8} {name:28} n={} max|diff|={max_abs:.4e} (rel {rel:.2e}) at {argmax}: cpu={:.5} plugin={:.5}  scale={scale:.3e} nan(cpu/plugin)={nan_a}/{nan_b}",
        a.len(),
        a.get(argmax).copied().unwrap_or(f32::NAN),
        b.get(argmax).copied().unwrap_or(f32::NAN)
    );
}

fn assign(ty: ElementType) -> Result<xla::XlaComputation> {
    let b = XlaBuilder::new("assign");
    let _old = b.parameter(0, ty, &[], "old")?;
    let new = b.parameter(1, ty, &[], "new")?;
    b.build(&new)
}

#[test]
fn plugin_parity() -> Result<()> {
    let Some((device_type, path)) = plugin_from_env() else {
        return Ok(());
    };
    let cpu = PjRtClient::cpu()?;
    let dev = PjRtClient::plugin(&device_type, &path)?;
    let (b, h, hd, c, d) = (4i64, 24i64, 128i64, 201i64, 3072i64);

    let mut cases: Vec<(&str, Vec<In>, Build)> = Vec::new();

    cases.push((
        "linear_bf16",
        vec![
            In::Bf16(vec![32, d], rand(1, (32 * d) as usize, 1.0)),
            In::Bf16(vec![d, 1024], rand(2, (d * 1024) as usize, 0.02)),
        ],
        Box::new(|_, p| p[0].dot_general(&p[1], &[1], &[0], &[], &[])),
    ));
    cases.push((
        "attention_bf16_scratch_mask",
        vec![
            In::Bf16(vec![b, h, 1, hd], rand(3, (b * h * hd) as usize, 1.0)),
            In::Bf16(vec![b, c, h, hd], rand(4, (b * c * h * hd) as usize, 1.0)),
            In::Bf16(vec![b, c, h, hd], rand(5, (b * c * h * hd) as usize, 1.0)),
        ],
        Box::new(move |bl, p| {
            let scale = bl.c0(1f32 / (hd as f32).sqrt())?.convert(PrimitiveType::Bf16)?;
            let attn = p[0].dot_general(&p[1], &[3], &[3], &[0, 1], &[0, 2])?.mul_(&scale)?; // [b,h,1,c]
                                                                                             // Ring-mask-like validity: slots > 150 and the scratch slot are -inf.
            let jj = bl.iota(ElementType::S32, &[b, 1, c], 2)?;
            let valid = jj.le(&bl.c0(150i32)?.broadcast(&[b, 1, c])?)?;
            let mask = valid.select(
                &bl.c0(0f32)?.broadcast(&[b, 1, c])?,
                &bl.c0(f32::NEG_INFINITY)?.broadcast(&[b, 1, c])?,
            )?;
            let mask =
                mask.convert(PrimitiveType::Bf16)?.broadcast_in_dim(&[b, h, 1, c], &[0, 2, 3])?;
            let probs = attn
                .add_(&mask)?
                .convert(PrimitiveType::F32)?
                .softmax(-1)?
                .convert(PrimitiveType::Bf16)?;
            probs.dot_general(&p[2], &[3], &[1], &[0, 1], &[0, 2]) // [b,h,1,hd]
        }),
    ));
    cases.push((
        "attention_bf16_f32_mask",
        vec![
            In::Bf16(vec![b, h, 1, hd], rand(3, (b * h * hd) as usize, 1.0)),
            In::Bf16(vec![b, c, h, hd], rand(4, (b * c * h * hd) as usize, 1.0)),
            In::Bf16(vec![b, c, h, hd], rand(5, (b * c * h * hd) as usize, 1.0)),
        ],
        Box::new(move |bl, p| {
            let scale = bl.c0(1f32 / (hd as f32).sqrt())?.convert(PrimitiveType::Bf16)?;
            let attn = p[0]
                .dot_general(&p[1], &[3], &[3], &[0, 1], &[0, 2])?
                .mul_(&scale)?
                .convert(PrimitiveType::F32)?;
            let jj = bl.iota(ElementType::S32, &[b, 1, c], 2)?;
            let valid = jj.le(&bl.c0(150i32)?.broadcast(&[b, 1, c])?)?;
            let mask = valid.select(
                &bl.c0(0f32)?.broadcast(&[b, 1, c])?,
                &bl.c0(f32::NEG_INFINITY)?.broadcast(&[b, 1, c])?,
            )?;
            let probs = attn
                .add_(&mask.broadcast_in_dim(&[b, h, 1, c], &[0, 2, 3])?)?
                .softmax(-1)?
                .convert(PrimitiveType::Bf16)?;
            probs.dot_general(&p[2], &[3], &[1], &[0, 1], &[0, 2])
        }),
    ));
    cases.push((
        "bf16_neg_inf_add",
        vec![In::Bf16(vec![8], rand(30, 8, 1.0))],
        Box::new(|bl, p| {
            let neg = bl.c0(f32::NEG_INFINITY)?.convert(PrimitiveType::Bf16)?.broadcast(&[8])?;
            let zero = bl.zero(ElementType::Bf16)?.broadcast(&[8])?;
            let jj = bl.iota(ElementType::S32, &[8], 0)?;
            let m = jj.lt(&bl.c0(4i32)?.broadcast(&[8])?)?.select(&zero, &neg)?;
            p[0].add_(&m)?.convert(PrimitiveType::F32)?.exp()
        }),
    ));
    cases.push((
        "softmax_masked_f32",
        vec![In::F32(vec![b, h, 1, c], rand(6, (b * h * c) as usize, 4.0))],
        Box::new(move |bl, p| {
            let jj = bl.iota(ElementType::S32, &[b, 1, c], 2)?;
            let valid = jj.le(&bl.c0(120i32)?.broadcast(&[b, 1, c])?)?;
            let mask = valid.select(
                &bl.c0(0f32)?.broadcast(&[b, 1, c])?,
                &bl.c0(f32::NEG_INFINITY)?.broadcast(&[b, 1, c])?,
            )?;
            p[0].add_(&mask.broadcast_in_dim(&[b, h, 1, c], &[0, 2, 3])?)?.softmax(-1)
        }),
    ));
    cases.push((
        "gelu_erf_f32",
        vec![In::F32(vec![16, 4096], rand(7, 16 * 4096, 3.0))],
        Box::new(|_, p| p[0].gelu_erf()),
    ));
    cases.push((
        "gelu_erf_bf16_roundtrip",
        vec![In::Bf16(vec![16, 4096], rand(8, 16 * 4096, 3.0))],
        Box::new(|_, p| p[0].convert(PrimitiveType::F32)?.gelu_erf()?.convert(PrimitiveType::Bf16)),
    ));
    cases.push((
        "onehot_lookup_bf16",
        vec![
            In::Bf16(vec![2049, 512], rand(9, 2049 * 512, 1.0)),
            In::S32(
                vec![32],
                (0..32).map(|i| if i % 7 == 3 { -1 } else { (i * 61) % 2049 }).collect(),
            ),
        ],
        Box::new(|bl, p| {
            let iota = bl.iota(ElementType::S32, &[32, 2049], 1)?;
            let onehot = iota
                .eq(&p[1].broadcast_in_dim(&[32, 2049], &[0])?)?
                .convert(PrimitiveType::Bf16)?;
            onehot.dot_general(&p[0], &[1], &[0], &[], &[])
        }),
    ));
    cases.push((
        "gather_lookup_bf16",
        vec![
            In::Bf16(vec![2049, 512], rand(9, 2049 * 512, 1.0)),
            In::S32(vec![32], (0..32).map(|i| (i * 61) % 2049).collect()),
        ],
        Box::new(|_, p| p[0].take(&p[1], 0)),
    ));
    cases.push((
        "rotary_bf16",
        vec![
            In::Bf16(vec![b, h, 1, hd], rand(10, (b * h * hd) as usize, 1.0)),
            In::F32(vec![1, 64], rand(11, 64, 1.0)),
            In::F32(vec![1, 64], rand(12, 64, 1.0)),
        ],
        Box::new(move |_, p| {
            let half = hd / 2;
            let x = p[0].reshape(&[b, h, 1, half, 2])?;
            let x0 = x.slice_in_dim1(0, 1, 4)?.reshape(&[b, h, 1, half])?;
            let x1 = x.slice_in_dim1(1, 2, 4)?.reshape(&[b, h, 1, half])?;
            let cos =
                p[1].convert(PrimitiveType::Bf16)?.broadcast_in_dim(&[b, h, 1, half], &[2, 3])?;
            let sin =
                p[2].convert(PrimitiveType::Bf16)?.broadcast_in_dim(&[b, h, 1, half], &[2, 3])?;
            let o0 = x0.mul_(&cos)?.sub_(&x1.mul_(&sin)?)?.reshape(&[b, h, 1, half, 1])?;
            let o1 = x0.mul_(&sin)?.add_(&x1.mul_(&cos)?)?.reshape(&[b, h, 1, half, 1])?;
            o0.concat_in_dim(&[&o1], 4)?.reshape(&[b, h, 1, hd])
        }),
    ));
    cases.push((
        "rmsnorm_f32_of_bf16",
        vec![
            In::Bf16(vec![32, 1, d], rand(13, (32 * d) as usize, 2.0)),
            In::Bf16(vec![d], rand(14, d as usize, 1.0)),
        ],
        Box::new(move |bl, p| {
            let x = p[0].convert(PrimitiveType::F32)?;
            let ms = x.mul_(&x)?.reduce_mean(&[2], true)?;
            let inv = ms.add_(&bl.c0(1e-8f32)?.broadcast(&[32, 1, 1])?)?.rsqrt()?;
            let w = p[1].convert(PrimitiveType::F32)?.broadcast_in_dim(&[32, 1, d], &[2])?;
            x.mul_(&inv.broadcast_in_dim(&[32, 1, d], &[0, 1, 2])?)?
                .mul_(&w)?
                .convert(PrimitiveType::Bf16)
        }),
    ));
    cases.push((
        "layernorm_f32",
        vec![
            In::F32(vec![32, 1024], rand(15, 32 * 1024, 2.0)),
            In::F32(vec![1024], rand(16, 1024, 1.0)),
            In::F32(vec![1024], rand(17, 1024, 0.1)),
        ],
        Box::new(|_, p| {
            p[0].layer_norm(
                -1,
                &p[1].broadcast_in_dim(&[32, 1024], &[1])?,
                &p[2].broadcast_in_dim(&[32, 1024], &[1])?,
            )
        }),
    ));
    cases.push((
        "scatter_ring_scratch_bf16",
        vec![
            In::Bf16(vec![4, 7, 2, 8], rand(18, 4 * 7 * 2 * 8, 1.0)),
            In::S32(vec![8, 2], vec![0, 1, 0, 2, 1, 6, 1, 6, 2, 5, 2, 0, 3, 3, 3, 4]),
            In::Bf16(vec![8, 2, 8], rand(19, 8 * 2 * 8, 1.0)),
        ],
        Box::new(|_, p| {
            p[0].scatter(&p[1], &p[2], &assign(ElementType::Bf16)?, &[1, 2], &[0, 1], &[0, 1], 1)
        }),
    ));
    cases.push((
        "ring_mask_multi_arith",
        vec![In::S32(vec![4], vec![0, 5, 199, 730])],
        Box::new(move |bl, p| {
            let (t, context) = (12i64, 200i64);
            let dims = [4, t, context];
            let ii = bl.iota(ElementType::S32, &dims, 1)?;
            let jj = bl.iota(ElementType::S32, &dims, 2)?;
            let pos = p[0].broadcast_in_dim(&dims, &[0])?;
            let q = pos.add_(&ii)?;
            let q_end = pos.add_(&bl.c0((t - 1) as i32)?.broadcast(&dims)?)?;
            let pj =
                q_end.sub_(&q_end.sub_(&jj)?.rem_(&bl.c0(context as i32)?.broadcast(&dims)?)?)?;
            let valid = pj.ge(&bl.c0(0i32)?.broadcast(&dims)?)?.and(&pj.le(&q)?)?;
            valid.select(&bl.c0(1f32)?.broadcast(&dims)?, &bl.c0(0f32)?.broadcast(&dims)?)
        }),
    ));
    cases.push((
        "conv1d_f32",
        vec![
            In::F32(vec![2, 64, 100], rand(20, 2 * 64 * 100, 1.0)),
            In::F32(vec![64, 64, 7], rand(21, 64 * 64 * 7, 0.05)),
        ],
        Box::new(|_, p| p[0].conv1d(&p[1], 1, 3, 1, 1)),
    ));
    cases.push((
        "conv_transpose1d_f32",
        vec![
            In::F32(vec![2, 64, 50], rand(22, 2 * 64 * 50, 1.0)),
            In::F32(vec![64, 64, 8], rand(23, 64 * 64 * 8, 0.05)),
        ],
        Box::new(|_, p| p[0].conv_transpose1d(&p[1], 4, 0, 0, 1, 1)),
    ));
    cases.push((
        "rev_f32",
        vec![In::F32(vec![3, 5, 7], rand(24, 105, 1.0))],
        Box::new(|_, p| p[0].rev(&[2])),
    ));
    cases.push((
        "elementwise_f32",
        vec![In::F32(vec![4096], rand(25, 4096, 3.0))],
        Box::new(|_, p| {
            let e = p[0].exp()?;
            let l = p[0].abs()?.add_(&e)?.log()?;
            p[0].logistic()?.add_(&l)?.add_(&p[0].tanh()?)?.add_(&p[0].mul_(&p[0])?.rsqrt()?)
        }),
    ));
    cases.push((
        "reductions_bf16",
        vec![In::Bf16(vec![32, 2048], rand(26, 32 * 2048, 2.0))],
        Box::new(|_, p| {
            let x = p[0].convert(PrimitiveType::F32)?;
            let s = x.reduce_sum(&[1], true)?;
            let m = x.reduce_max(&[1], true)?;
            let am =
                x.argmax(ElementType::S32, -1)?.reshape(&[32, 1])?.convert(PrimitiveType::F32)?;
            s.concat_in_dim(&[&m, &am], 1)
        }),
    ));
    cases.push((
        "layout_roundtrip_bf16",
        vec![In::Bf16(vec![32, 1, h, hd], rand(27, (32 * h * hd) as usize, 1.0))],
        Box::new(move |_, p| {
            let t = p[0].transpose(&[0, 2, 1, 3])?.reshape(&[32, h, hd])?;
            t.reshape(&[32, h, 1, hd])?.transpose(&[0, 2, 1, 3])
        }),
    ));

    for (name, inputs, build) in cases.iter() {
        let a = match run(&cpu, name, inputs, build) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("CPU FAIL {name}: {e}");
                continue;
            }
        };
        match run(&dev, name, inputs, build) {
            Ok(v) => compare(name, &a, &v),
            Err(e) => {
                eprintln!("PLUGIN FAIL {name}: {}", e.to_string().lines().next().unwrap_or(""))
            }
        }
    }
    // Uniform sampling statistics (no CPU reference: different generators).
    {
        let builder = XlaBuilder::new("rng");
        let rng = builder.parameter(0, ElementType::U64, &[2], "rng")?;
        let (_s, u) = rng.sample_uniform(
            xla::RandomAlgorithm::Default,
            ElementType::F32,
            &[1024, 1024],
            0.0,
            1.0,
        )?;
        let exe = dev.compile(&u.build()?)?;
        let st = dev.buffer_from_host_buffer(&[42u64, 7u64], &[2], None)?;
        let u = exe.execute_b(&[&st])?[0][0].to_literal_sync()?.to_vec::<f32>()?;
        let n = u.len() as f64;
        let mean = u.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = u.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
        let (lo, hi) = (
            u.iter().cloned().fold(f32::MAX, f32::min),
            u.iter().cloned().fold(f32::MIN, f32::max),
        );
        let distinct = {
            let mut s = u.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            s.sort_unstable();
            s.dedup();
            s.len()
        };
        eprintln!("rng_uniform_default: mean={mean:.4} (0.5) var={var:.4} (0.0833) range=[{lo:.3e},{hi:.4}] distinct={distinct}/{}", u.len());
    }
    Ok(())
}
