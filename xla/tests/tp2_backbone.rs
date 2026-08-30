//! Tensor-parallel (2 replicas) tests through a PJRT plugin, gated on
//! `XLA_PJRT_PLUGIN`. `tp2_backbone_step` measures the TTS backbone shape at
//! fb=16 with Megatron-style sharding: 12 of 24 heads and 4224 of 8448 MLP
//! hidden per core, cross-replica all-reduce after each attention output and
//! MLP output, per-replica ring KV caches, weights as resident device buffers.
use xla::{ElementType, PjRtBuffer, PjRtClient, PrimitiveType, XlaBuilder, XlaComputation, XlaOp};

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

/// Overwrite-combiner for the ring-cache scatter (new value wins).
fn assign_computation(ty: ElementType) -> xla::Result<XlaComputation> {
    let b = XlaBuilder::new("assign");
    let _old = b.parameter(0, ty, &[], "old")?;
    let new = b.parameter(1, ty, &[], "new")?;
    new.build()
}

#[test]
fn tp2_all_reduce_smoke() -> xla::Result<()> {
    let Some(client) = plugin_client() else { return Ok(()) };
    let builder = XlaBuilder::new("tp2-smoke");
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
    eprintln!("all_reduce smoke ok: {r0:?}");
    Ok(())
}

const B: i64 = 16; // fb at batch 8 (CFG-doubled)
const D: i64 = 3072;
const HD: i64 = 128;
const L: usize = 16;
const C: i64 = 200;
const TKV: i64 = 375;

/// TP degree from `TP_DEGREE` (default 2): per-replica heads and MLP hidden.
fn tp() -> i64 {
    std::env::var("TP_DEGREE").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
}
fn hl() -> i64 {
    24 / tp()
}
fn ffl() -> i64 {
    8448 / tp()
}

fn bf16_buffer(
    client: &PjRtClient,
    dims: &[i64],
    seed: u32,
    dev: &xla::PjRtDevice,
) -> xla::Result<PjRtBuffer> {
    // Cheap patterned bf16 data (parameters are never constant-folded, so the
    // pattern only needs to be numerically tame).
    let n: usize = dims.iter().product::<i64>() as usize;
    let mut bytes = Vec::with_capacity(2 * n);
    let mut s = seed | 1;
    for _ in 0..n {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((s >> 16) as f32 / 65536.0 - 0.5) * 0.05;
        bytes.extend_from_slice(&v.to_bits().to_le_bytes()[2..4]);
    }
    let dims_u: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    client.buffer_from_host_raw_bytes(ElementType::Bf16, &bytes, &dims_u, Some(dev))
}

fn rmsnorm(x: &XlaOp) -> xla::Result<XlaOp> {
    let xf = x.convert(PrimitiveType::F32)?;
    let mean = xf.mul_(&xf)?.reduce_mean(&[1], true)?;
    let b = x.builder();
    let inv = b
        .c0(1f32)?
        .broadcast(&[B, 1])?
        .div_(&mean.add_(&b.c0(1e-8f32)?.broadcast(&[B, 1])?)?.sqrt()?)?;
    xf.mul_(&inv.broadcast_in_dim(&[B, D], &[0, 1])?)?.convert(PrimitiveType::Bf16)
}

fn rope(x: &XlaOp, cos: &XlaOp, sin: &XlaOp) -> xla::Result<XlaOp> {
    // x [B, hl(), HD]; cos/sin [B, 1, HD/2]
    let half = HD / 2;
    let xr = x.reshape(&[B, hl(), half, 2])?;
    let x0 = xr.slice_in_dim1(0, 1, 3)?.reshape(&[B, hl(), half])?;
    let x1 = xr.slice_in_dim1(1, 2, 3)?.reshape(&[B, hl(), half])?;
    let cos = cos.broadcast_in_dim(&[B, hl(), half], &[0, 1, 2])?;
    let sin = sin.broadcast_in_dim(&[B, hl(), half], &[0, 1, 2])?;
    let y0 = x0.mul_(&cos)?.sub_(&x1.mul_(&sin)?)?;
    let y1 = x0.mul_(&sin)?.add_(&x1.mul_(&cos)?)?;
    let y =
        y0.reshape(&[B, hl(), half, 1])?.concat_in_dim(&[y1.reshape(&[B, hl(), half, 1])?], 3)?;
    y.reshape(&[B, hl(), HD])
}

#[test]
fn tp2_backbone_step() -> xla::Result<()> {
    let Some(client) = plugin_client() else { return Ok(()) };
    let ty = ElementType::Bf16;
    let dt = PrimitiveType::Bf16;

    // ---- graph -------------------------------------------------------
    let b = XlaBuilder::new("tp2-backbone-step");
    let pc = std::cell::Cell::new(0i64);
    let param = |t: ElementType, dims: &[i64], name: &str| {
        let op = b.parameter(pc.get(), t, dims, name);
        pc.set(pc.get() + 1);
        op
    };
    let x_in = param(ty, &[B, D], "x")?;
    let pos = param(ElementType::S32, &[B], "pos")?;
    let mut weight_ops = Vec::new();
    for l in 0..L {
        let names = [
            ("wqkv", vec![D, 3 * hl() * HD]),
            ("wo", vec![hl() * HD, D]),
            ("wq_x", vec![D, hl() * HD]),
            ("wo_x", vec![hl() * HD, D]),
            ("w_in", vec![D, 2 * ffl()]),
            ("w_out", vec![ffl(), D]),
            ("xk", vec![B, TKV, hl(), HD]),
            ("xv", vec![B, TKV, hl(), HD]),
        ];
        let mut ops = Vec::new();
        for (n, dims) in names {
            ops.push(param(ty, &dims, &format!("{n}_{l}"))?);
        }
        weight_ops.push(ops);
    }
    let first_state = pc.get();
    let mut cache_ops = Vec::new();
    for l in 0..L {
        let kc = param(ty, &[B, C, hl(), HD], &format!("kc_{l}"))?;
        let vc = param(ty, &[B, C, hl(), HD], &format!("vc_{l}"))?;
        cache_ops.push((kc, vc));
    }

    // rope tables + ring mask from pos
    let half = HD / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / 10000f32.powf(i as f32 / half as f32)).collect();
    let inv = b.constant_r1(&inv)?.reshape(&[1, 1, half])?;
    let posf = pos.convert(PrimitiveType::F32)?.reshape(&[B, 1, 1])?;
    let ang = posf.mul_(&inv.broadcast_in_dim(&[B, 1, half], &[0, 1, 2])?)?;
    let (cos, sin) = (ang.cos()?.convert(dt)?, ang.sin()?.convert(dt)?);
    let iota_c = b.iota(ElementType::S32, &[B, C], 1)?;
    let valid = iota_c.le(&pos.reshape(&[B, 1])?.broadcast_in_dim(&[B, C], &[0, 1])?)?;
    let zeros = b.c0(0f32)?.broadcast(&[B, C])?;
    let neg = b.c0(f32::NEG_INFINITY)?.broadcast(&[B, C])?;
    let mask = valid.select(&zeros, &neg)?;
    let slot = pos.rem_(&b.c0(C as i32)?.broadcast(&[B])?)?;
    let iota_b = b.iota(ElementType::S32, &[B], 0)?;
    let scatter_idx = iota_b.reshape(&[B, 1])?.concat_in_dim(&[slot.reshape(&[B, 1])?], 1)?;
    let assign = assign_computation(ty)?;
    let addf = add_computation(ElementType::F32)?;
    let scale = b.c0(1f32 / (HD as f32).sqrt())?;

    let mut x = x_in.clone();
    let mut new_caches = Vec::new();
    for l in 0..L {
        let w = &weight_ops[l];
        let (kc_in, vc_in) = &cache_ops[l];
        // self-attention over local heads
        let h = rmsnorm(&x)?;
        let qkv = h.dot(&w[0])?.reshape(&[B, 3, hl(), HD])?;
        let q = qkv.slice_in_dim1(0, 1, 1)?.reshape(&[B, hl(), HD])?;
        let k = qkv.slice_in_dim1(1, 2, 1)?.reshape(&[B, hl(), HD])?;
        let v = qkv.slice_in_dim1(2, 3, 1)?.reshape(&[B, hl(), HD])?;
        let (q, k) = (rope(&q, &cos, &sin)?, rope(&k, &cos, &sin)?);
        let scatter = |cache: &XlaOp, new: &XlaOp| -> xla::Result<XlaOp> {
            cache.scatter(
                &scatter_idx,
                &new.reshape(&[B, hl(), HD])?,
                &assign,
                &[1, 2],
                &[0, 1],
                &[0, 1],
                1,
            )
        };
        let kc = scatter(kc_in, &k)?;
        let vc = scatter(vc_in, &v)?;
        // q [B,hl(),HD] x kc [B,C,hl(),HD] -> [B,hl(),C], contracting HD, batch (B, hl())
        let attn = q.dot_general(&kc, &[2], &[3], &[0, 1], &[0, 2])?;
        let attn = attn
            .convert(PrimitiveType::F32)?
            .mul_(&scale.broadcast(&[B, hl(), C])?)?
            .add_(&mask.reshape(&[B, 1, C])?.broadcast_in_dim(&[B, hl(), C], &[0, 1, 2])?)?;
        let attn = attn.softmax(2)?.convert(dt)?;
        let a = attn.dot_general(&vc, &[2], &[1], &[0, 1], &[0, 2])?; // [B,hl(),HD]
        let o = a.reshape(&[B, hl() * HD])?.dot(&w[1])?;
        let red = if tp() > 1 {
            o.convert(PrimitiveType::F32)?.all_reduce(&addf)?
        } else {
            o.convert(PrimitiveType::F32)?
        };
        x = x.add_(&red.convert(dt)?)?;
        // cross-attention over local heads (static kv)
        let h = rmsnorm(&x)?;
        let qx = h.dot(&w[2])?.reshape(&[B, hl(), HD])?;
        let attn = qx.dot_general(&w[6], &[2], &[3], &[0, 1], &[0, 2])?;
        let attn = attn
            .convert(PrimitiveType::F32)?
            .mul_(&scale.broadcast(&[B, hl(), TKV])?)?
            .softmax(2)?
            .convert(dt)?;
        let a = attn.dot_general(&w[7], &[2], &[1], &[0, 1], &[0, 2])?;
        let o = a.reshape(&[B, hl() * HD])?.dot(&w[3])?;
        let red = if tp() > 1 {
            o.convert(PrimitiveType::F32)?.all_reduce(&addf)?
        } else {
            o.convert(PrimitiveType::F32)?
        };
        x = x.add_(&red.convert(dt)?)?;
        // gated MLP over local hidden
        let h = rmsnorm(&x)?;
        let g = h.dot(&w[4])?;
        let ga = g.slice_in_dim1(0, ffl(), 1)?;
        let gb = g.slice_in_dim1(ffl(), 2 * ffl(), 1)?;
        let silu = ga
            .convert(PrimitiveType::F32)?
            .logistic()?
            .mul_(&ga.convert(PrimitiveType::F32)?)?
            .convert(dt)?;
        let m = silu.mul_(&gb)?.dot(&w[5])?;
        let red = if tp() > 1 {
            m.convert(PrimitiveType::F32)?.all_reduce(&addf)?
        } else {
            m.convert(PrimitiveType::F32)?
        };
        x = x.add_(&red.convert(dt)?)?;
        new_caches.push((kc, vc));
    }
    let xf32 = x.convert(PrimitiveType::F32)?;
    let mut outs = vec![x];
    for (kc, vc) in &new_caches {
        outs.push(kc.clone());
        outs.push(vc.clone());
    }
    outs.push(xf32);
    // alias cache outputs (1..) to cache params so updates are in place
    for i in 0..(2 * L) as i64 {
        b.setup_alias(1 + i, first_state + i);
    }
    let computation = b.tuple(&outs.iter().collect::<Vec<_>>())?.build()?;
    let exe = client.compile_replicated(&computation, tp() as usize)?;
    eprintln!("compiled");

    // ---- buffers per replica -----------------------------------------
    let devs = client.addressable_devices();
    let mut weights: Vec<Vec<PjRtBuffer>> = Vec::new();
    let mut caches: Vec<Vec<PjRtBuffer>> = Vec::new();
    for (r, dev) in devs.iter().take(tp() as usize).enumerate() {
        let mut ws = Vec::new();
        for l in 0..L {
            for (i, dims) in [
                vec![D, 3 * hl() * HD],
                vec![hl() * HD, D],
                vec![D, hl() * HD],
                vec![hl() * HD, D],
                vec![D, 2 * ffl()],
                vec![ffl(), D],
                vec![B, TKV, hl(), HD],
                vec![B, TKV, hl(), HD],
            ]
            .iter()
            .enumerate()
            {
                ws.push(bf16_buffer(&client, dims, (r * 1000 + l * 10 + i) as u32, dev)?);
            }
        }
        weights.push(ws);
        let mut cs = Vec::new();
        for _ in 0..(2 * L) {
            let n = (B * C * hl() * HD) as usize;
            cs.push(client.buffer_from_host_raw_bytes(
                ElementType::Bf16,
                &vec![0u8; 2 * n],
                &[B as usize, C as usize, hl() as usize, HD as usize],
                Some(dev),
            )?);
        }
        caches.push(cs);
    }
    let x_host: Vec<f32> = (0..(B * D) as usize).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect();

    let mut step_no = 0i32;
    let mut xbufs: Vec<PjRtBuffer> = Vec::new();
    let last_f32: std::cell::RefCell<Vec<PjRtBuffer>> = std::cell::RefCell::new(Vec::new());
    let run = |caches: &mut Vec<Vec<PjRtBuffer>>,
               xbufs: &mut Vec<PjRtBuffer>,
               step_no: i32|
     -> xla::Result<()> {
        let pos_host = vec![step_no; B as usize];
        let pos_bufs: Vec<PjRtBuffer> = devs
            .iter()
            .take(tp() as usize)
            .map(|dev| client.buffer_from_host_buffer(&pos_host, &[B as usize], Some(dev)))
            .collect::<xla::Result<_>>()?;
        if xbufs.is_empty() {
            let n = (B * D) as usize;
            let mut bytes = Vec::with_capacity(2 * n);
            for v in &x_host {
                bytes.extend_from_slice(&v.to_bits().to_le_bytes()[2..4]);
            }
            for dev in devs.iter().take(tp() as usize) {
                xbufs.push(client.buffer_from_host_raw_bytes(
                    ElementType::Bf16,
                    &bytes,
                    &[B as usize, D as usize],
                    Some(dev),
                )?);
            }
        }
        let mut all: Vec<Vec<&PjRtBuffer>> = Vec::new();
        for r in 0..tp() as usize {
            let mut a: Vec<&PjRtBuffer> = vec![&xbufs[r], &pos_bufs[r]];
            a.extend(weights[r].iter());
            a.extend(caches[r].iter());
            all.push(a);
        }
        let out = exe.execute_replicated_b(&all)?;
        let mut new_x = Vec::new();
        let mut new_c: Vec<Vec<PjRtBuffer>> = Vec::new();
        let mut xf32 = Vec::new();
        for mut outs in out.into_iter() {
            new_x.push(outs.remove(0));
            xf32.push(outs.pop().expect("f32 copy output"));
            new_c.push(outs);
        }
        *xbufs = new_x;
        *caches = new_c;
        *last_f32.borrow_mut() = xf32;
        Ok(())
    };

    run(&mut caches, &mut xbufs, step_no)?;
    eprintln!("first step ok");
    for _ in 0..5 {
        step_no += 1;
        run(&mut caches, &mut xbufs, step_no)?;
    }
    let iters = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        step_no += 1;
        run(&mut caches, &mut xbufs, step_no)?;
    }
    // sync
    let _ = xbufs[0].to_literal_sync()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!("TP={} backbone step through xla-rs: {ms:.1} ms over {iters} iters", tp());
    let lf = last_f32.borrow();
    let xf = lf[0].to_literal_sync()?.to_vec::<f32>()?;
    let finite = xf.iter().all(|v| v.is_finite());
    eprintln!("finite: {finite}");
    assert!(finite);
    Ok(())
}
