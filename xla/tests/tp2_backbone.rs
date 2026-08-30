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
/// With `XLA_TP_EXTRAS=1`, the backbone test also carries the worker's
/// embedding/head extras: 32 one-hot audio embeddings ([2049, D] each), the
/// text embedding ([20001, D] through the out1/out2 pair) and the 4-class
/// text head with an argmax readout — ~563 MB of extra weight stream.
fn extras() -> bool {
    std::env::var("XLA_TP_EXTRAS").is_ok_and(|v| v == "1")
}
const AUDIO_VOCAB: i64 = 2049;
const N_AUDIO_EMB: usize = 32;
const TEXT_VOCAB: i64 = 20001;
const TEXT_OUT: i64 = 4;

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
    let mut extra_ops = Vec::new();
    if extras() {
        extra_ops.push(param(ElementType::S32, &[B, N_AUDIO_EMB as i64], "audio_tokens")?);
        extra_ops.push(param(ElementType::S32, &[B], "text_tok1")?);
        extra_ops.push(param(ElementType::S32, &[B], "text_tok2")?);
        for i in 0..N_AUDIO_EMB {
            extra_ops.push(param(ty, &[AUDIO_VOCAB, D], &format!("audio_emb_{i}"))?);
        }
        extra_ops.push(param(ty, &[TEXT_VOCAB, D], "text_emb")?);
        extra_ops.push(param(ty, &[D, D], "text_out1")?);
        extra_ops.push(param(ty, &[D, D], "text_out2")?);
        extra_ops.push(param(ty, &[D, TEXT_OUT], "text_head")?);
    }
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
    if extras() {
        let onehot = |tok: &XlaOp, vocab: i64| -> xla::Result<XlaOp> {
            let iota = b.iota(ElementType::S32, &[B, vocab], 1)?;
            iota.eq(&tok.reshape(&[B, 1])?.broadcast_in_dim(&[B, vocab], &[0, 1])?)?.convert(dt)
        };
        let audio_tokens = &extra_ops[0];
        for i in 0..N_AUDIO_EMB {
            let tok = audio_tokens.slice_in_dim1(i as i64, i as i64 + 1, 1)?.reshape(&[B])?;
            x = x.add_(&onehot(&tok, AUDIO_VOCAB)?.dot(&extra_ops[3 + i])?)?;
        }
        let te = &extra_ops[3 + N_AUDIO_EMB];
        let e1 = onehot(&extra_ops[1], TEXT_VOCAB)?.dot(te)?.dot(&extra_ops[4 + N_AUDIO_EMB])?;
        let e2 = onehot(&extra_ops[2], TEXT_VOCAB)?.dot(te)?.dot(&extra_ops[5 + N_AUDIO_EMB])?;
        x = x.add_(&e1)?.add_(&e2)?;
    }
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
    let xf32 = if extras() {
        let logits = rmsnorm(&x)?.dot(&extra_ops[6 + N_AUDIO_EMB])?.convert(PrimitiveType::F32)?;
        let action = logits.argmax(ElementType::S32, 1)?;
        // fold the sampled action back into the f32 readout so nothing is DCE'd
        x.convert(PrimitiveType::F32)?.add_(
            &action
                .convert(PrimitiveType::F32)?
                .reshape(&[B, 1])?
                .broadcast_in_dim(&[B, D], &[0, 1])?
                .mul_(&b.c0(1e-6f32)?.broadcast(&[B, D])?)?,
        )?
    } else {
        x.convert(PrimitiveType::F32)?
    };
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
    let mut extra_bufs: Vec<Vec<PjRtBuffer>> = Vec::new();
    for (r, dev) in devs.iter().take(tp() as usize).enumerate() {
        if extras() {
            let mut es = Vec::new();
            es.push(client.buffer_from_host_buffer(
                &vec![5i32; (B as usize) * N_AUDIO_EMB],
                &[B as usize, N_AUDIO_EMB],
                Some(dev),
            )?);
            for _ in 0..2 {
                es.push(client.buffer_from_host_buffer(
                    &vec![7i32; B as usize],
                    &[B as usize],
                    Some(dev),
                )?);
            }
            for i in 0..N_AUDIO_EMB {
                es.push(bf16_buffer(
                    &client,
                    &[AUDIO_VOCAB, D],
                    (7_000_000 + r * 1000 + i) as u32,
                    dev,
                )?);
            }
            es.push(bf16_buffer(&client, &[TEXT_VOCAB, D], (7_100_000 + r) as u32, dev)?);
            es.push(bf16_buffer(&client, &[D, D], (7_200_000 + r) as u32, dev)?);
            es.push(bf16_buffer(&client, &[D, D], (7_300_000 + r) as u32, dev)?);
            es.push(bf16_buffer(&client, &[D, TEXT_OUT], (7_400_000 + r) as u32, dev)?);
            extra_bufs.push(es);
        } else {
            extra_bufs.push(Vec::new());
        }
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
            a.extend(extra_bufs[r].iter());
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
    eprintln!(
        "TP={} extras={} backbone step through xla-rs: {ms:.1} ms over {iters} iters",
        tp(),
        extras()
    );
    let lf = last_f32.borrow();
    let xf = lf[0].to_literal_sync()?.to_vec::<f32>()?;
    let finite = xf.iter().all(|v| v.is_finite());
    eprintln!("finite: {finite}");
    assert!(finite);
    Ok(())
}

// ---- depformer ---------------------------------------------------------
// One TTS frame runs the whole 31-slice chain as a single graph: each slice
// has its own weights (4 layers, d=1024, 16 heads x 64, gated-SiLU ff 3072),
// attends over the k/v of all previous slices (in-graph concat, no external
// cache), and feeds its argmax token to the next slice's embedding. TP shards
// heads and MLP hidden; 2 all-reduces per layer, 8 per slice, 248 per step.
const DD: i64 = 1024;
const DHD: i64 = 64;
const DL: usize = 4;
const NSLICES: usize = 31;
const VOCAB: i64 = 2049;

fn dhl() -> i64 {
    16 / tp()
}
fn dffl() -> i64 {
    3072 / tp()
}

fn df_rmsnorm(x: &XlaOp) -> xla::Result<XlaOp> {
    let xf = x.convert(PrimitiveType::F32)?;
    let mean = xf.mul_(&xf)?.reduce_mean(&[1], true)?;
    let b = x.builder();
    let inv = b
        .c0(1f32)?
        .broadcast(&[B, 1])?
        .div_(&mean.add_(&b.c0(1e-8f32)?.broadcast(&[B, 1])?)?.sqrt()?)?;
    xf.mul_(&inv.broadcast_in_dim(&[B, DD], &[0, 1])?)?.convert(PrimitiveType::Bf16)
}

#[test]
fn tp2_depformer_step() -> xla::Result<()> {
    let Some(client) = plugin_client() else { return Ok(()) };
    let ty = ElementType::Bf16;
    let dt = PrimitiveType::Bf16;

    let b = XlaBuilder::new("tp2-depformer-step");
    let pc = std::cell::Cell::new(0i64);
    let param = |t: ElementType, dims: &[i64], name: &str| {
        let op = b.parameter(pc.get(), t, dims, name);
        pc.set(pc.get() + 1);
        op
    };
    let ys = param(ty, &[B, D], "ys")?; // backbone hidden
    let text = param(ElementType::S32, &[B], "text")?;
    // per-slice weights: linear_in, emb, linear_out, then per layer
    // wqkv/wo/w_in/w_out over local heads / local hidden
    let mut slice_ops = Vec::new();
    for s in 0..NSLICES {
        let mut ops = Vec::new();
        for (n, dims) in
            [("lin_in", vec![D, DD]), ("emb", vec![VOCAB, DD]), ("lin_out", vec![DD, VOCAB - 1])]
        {
            ops.push(param(ty, &dims, &format!("{n}_{s}"))?);
        }
        for l in 0..DL {
            for (n, dims) in [
                ("wqkv", vec![DD, 3 * dhl() * DHD]),
                ("wo", vec![dhl() * DHD, DD]),
                ("w_in", vec![DD, 2 * dffl()]),
                ("w_out", vec![dffl(), DD]),
            ] {
                ops.push(param(ty, &dims, &format!("{n}_{s}_{l}"))?);
            }
        }
        slice_ops.push(ops);
    }
    let addf = add_computation(ElementType::F32)?;
    let scale = b.c0(1f32 / (DHD as f32).sqrt())?;

    let mut ks: Vec<Vec<XlaOp>> = vec![Vec::new(); DL];
    let mut vs: Vec<Vec<XlaOp>> = vec![Vec::new(); DL];
    let mut tokens = text.clone();
    let mut all_tokens = Vec::new();
    let mut last_hidden = ys.clone();
    for s in 0..NSLICES {
        let w = &slice_ops[s];
        let xs = ys.dot(&w[0])?; // [B, DD]
                                 // one-hot embedding matmul (gathers are slow on neuron, cf
                                 // XLA_MOSHI_EMBEDDING_ONEHOT in the worker)
        let iota_v = b.iota(ElementType::S32, &[B, VOCAB], 1)?;
        let oh = iota_v
            .eq(&tokens.reshape(&[B, 1])?.broadcast_in_dim(&[B, VOCAB], &[0, 1])?)?
            .convert(dt)?;
        let emb = oh.dot(&w[1])?; // [B, DD]
        let mut x = xs.add_(&emb)?;
        for l in 0..DL {
            let h = df_rmsnorm(&x)?;
            let qkv = h.dot(&w[3 + 4 * l])?.reshape(&[B, 3, dhl(), DHD])?;
            let q = qkv.slice_in_dim1(0, 1, 1)?.reshape(&[B, dhl(), DHD])?;
            let k = qkv.slice_in_dim1(1, 2, 1)?.reshape(&[B, dhl(), 1, DHD])?;
            let v = qkv.slice_in_dim1(2, 3, 1)?.reshape(&[B, dhl(), 1, DHD])?;
            ks[l].push(k);
            vs[l].push(v);
            let t = ks[l].len() as i64;
            let kcat = if t == 1 {
                ks[l][0].clone()
            } else {
                ks[l][0].concat_in_dim(&ks[l][1..].iter().collect::<Vec<_>>(), 2)?
            };
            let vcat = if t == 1 {
                vs[l][0].clone()
            } else {
                vs[l][0].concat_in_dim(&vs[l][1..].iter().collect::<Vec<_>>(), 2)?
            };
            // q [B,h,HD] x k [B,h,t,HD] -> [B,h,t]
            let attn = q.dot_general(&kcat, &[2], &[3], &[0, 1], &[0, 1])?;
            let attn = attn
                .convert(PrimitiveType::F32)?
                .mul_(&scale.broadcast(&[B, dhl(), t])?)?
                .softmax(2)?
                .convert(dt)?;
            let a = attn.dot_general(&vcat, &[2], &[2], &[0, 1], &[0, 1])?; // [B,h,HD]
            let o = a.reshape(&[B, dhl() * DHD])?.dot(&w[4 + 4 * l])?;
            let red = if tp() > 1 {
                o.convert(PrimitiveType::F32)?.all_reduce(&addf)?
            } else {
                o.convert(PrimitiveType::F32)?
            };
            x = x.add_(&red.convert(dt)?)?;
            let h = df_rmsnorm(&x)?;
            let g = h.dot(&w[5 + 4 * l])?;
            let ga = g.slice_in_dim1(0, dffl(), 1)?;
            let gb = g.slice_in_dim1(dffl(), 2 * dffl(), 1)?;
            let silu = ga
                .convert(PrimitiveType::F32)?
                .logistic()?
                .mul_(&ga.convert(PrimitiveType::F32)?)?
                .convert(dt)?;
            let m = silu.mul_(&gb)?.dot(&w[6 + 4 * l])?;
            let red = if tp() > 1 {
                m.convert(PrimitiveType::F32)?.all_reduce(&addf)?
            } else {
                m.convert(PrimitiveType::F32)?
            };
            x = x.add_(&red.convert(dt)?)?;
        }
        let logits = df_rmsnorm(&x)?.dot(&w[2])?.convert(PrimitiveType::F32)?; // [B, VOCAB-1]
        tokens = logits.argmax(ElementType::S32, 1)?;
        all_tokens.push(tokens.reshape(&[B, 1])?);
        last_hidden = x;
    }
    let codes = all_tokens[0].concat_in_dim(&all_tokens[1..].iter().collect::<Vec<_>>(), 1)?;
    let hf32 = last_hidden.convert(PrimitiveType::F32)?;
    let computation = b.tuple(&[&codes, &hf32])?.build()?;
    let exe = client.compile_replicated(&computation, tp() as usize)?;
    eprintln!("compiled");

    let devs = client.addressable_devices();
    let mut weights: Vec<Vec<PjRtBuffer>> = Vec::new();
    for (r, dev) in devs.iter().take(tp() as usize).enumerate() {
        let mut ws = Vec::new();
        for s in 0..NSLICES {
            for (i, dims) in [vec![D, DD], vec![VOCAB, DD], vec![DD, VOCAB - 1]].iter().enumerate()
            {
                ws.push(bf16_buffer(&client, dims, (r * 100_000 + s * 100 + i) as u32, dev)?);
            }
            for l in 0..DL {
                for (i, dims) in [
                    vec![DD, 3 * dhl() * DHD],
                    vec![dhl() * DHD, DD],
                    vec![DD, 2 * dffl()],
                    vec![dffl(), DD],
                ]
                .iter()
                .enumerate()
                {
                    ws.push(bf16_buffer(
                        &client,
                        dims,
                        (r * 100_000 + s * 100 + 10 + l * 4 + i) as u32,
                        dev,
                    )?);
                }
            }
        }
        weights.push(ws);
    }
    let ys_host: Vec<f32> = (0..(B * D) as usize).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect();
    let mut ys_bytes = Vec::with_capacity(2 * ys_host.len());
    for v in &ys_host {
        ys_bytes.extend_from_slice(&v.to_bits().to_le_bytes()[2..4]);
    }
    let mut ys_bufs = Vec::new();
    let mut text_bufs = Vec::new();
    for dev in devs.iter().take(tp() as usize) {
        ys_bufs.push(client.buffer_from_host_raw_bytes(
            ElementType::Bf16,
            &ys_bytes,
            &[B as usize, D as usize],
            Some(dev),
        )?);
        text_bufs.push(client.buffer_from_host_buffer(
            &vec![3i32; B as usize],
            &[B as usize],
            Some(dev),
        )?);
    }
    let run = || -> xla::Result<Vec<Vec<PjRtBuffer>>> {
        let mut all: Vec<Vec<&PjRtBuffer>> = Vec::new();
        for r in 0..tp() as usize {
            let mut a: Vec<&PjRtBuffer> = vec![&ys_bufs[r], &text_bufs[r]];
            a.extend(weights[r].iter());
            all.push(a);
        }
        exe.execute_replicated_b(&all)
    };
    let out = run()?;
    eprintln!("first step ok");
    for _ in 0..5 {
        run()?;
    }
    let iters = 50;
    let t0 = std::time::Instant::now();
    let mut last = None;
    for _ in 0..iters {
        last = Some(run()?);
    }
    let last = last.unwrap();
    let _ = last[0][0].to_literal_sync()?;
    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    eprintln!("TP={} depformer step through xla-rs: {ms:.1} ms over {iters} iters", tp());
    let hf = out[0][1].to_literal_sync()?.to_vec::<f32>()?;
    let finite = hf.iter().all(|v| v.is_finite());
    eprintln!("finite: {finite}");
    assert!(finite);
    Ok(())
}
