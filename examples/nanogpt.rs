// A very simple GPT implementation based on https://github.com/karpathy/nanoGPT
// This only contains the inference part as the xla crate does not support backpropagation.
// No dropout as this is inference only.
use anyhow::{Context, Result};
use std::collections::HashMap;

extern crate xla;
use xla::{Literal, XlaBuilder, XlaOp};

const ET: xla::PrimitiveType = xla::PrimitiveType::F32;

#[derive(Clone)]
struct VarStore {
    path: Vec<String>,
    weights: std::rc::Rc<std::cell::RefCell<HashMap<String, Literal>>>,
}

impl VarStore {
    fn new<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let weights = xla::Literal::read_npz(path)?;
        let weights = weights.into_iter().collect::<HashMap<_, _>>();
        let weights = std::rc::Rc::new(std::cell::RefCell::new(weights));
        Ok(VarStore { path: vec![], weights })
    }

    fn len(&self) -> usize {
        self.weights.borrow().len()
    }

    fn take(&mut self, s: &str, expected_dims: &[usize]) -> Result<Literal> {
        let path = format!("{}.{s}", self.path.join("."));
        let literal = self
            .weights
            .borrow_mut()
            .remove(&path)
            .with_context(|| format!("cannot find {path} in VarStore"))?;
        let shape = literal.shape()?;
        let element_type = shape.element_type();
        let dims = shape.dimensions();
        if element_type != ET {
            anyhow::bail!(
                "unexpected element type for {}, got {:?} expected {:?}",
                path,
                element_type,
                ET
            )
        }
        if dims.iter().zip(expected_dims.iter()).any(|(u, v)| *u != *v as i64) {
            anyhow::bail!(
                "unexpected dims for {}, got {:?} expected {:?}",
                path,
                dims,
                expected_dims
            )
        }
        Ok(literal)
    }
}

impl<S: ToString> std::ops::Div<S> for &VarStore {
    type Output = VarStore;

    fn div(self, rhs: S) -> VarStore {
        let mut path = self.path.clone();
        path.push(rhs.to_string());
        VarStore { path, weights: self.weights.clone() }
    }
}

impl<S: ToString> std::ops::Div<S> for VarStore {
    type Output = VarStore;

    fn div(self, rhs: S) -> VarStore {
        &self / rhs
    }
}

fn new_gelu(x: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let sqrt_two_over_pi = b.c0((2f32 / std::f32::consts::PI).sqrt());
    // 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    let v = (sqrt_two_over_pi * ((b.c0(0.044715f32) * x.pow(&b.c0(3f32))?)? + x)?)?;
    let res = ((b.c0(0.5f32) * x)? * (v.tanh()? + b.c0(1f32))?)?;
    Ok(res)
}

struct Embedding {
    embeddings: Literal,
}

impl Embedding {
    fn new(mut vs: VarStore, vocab_size: usize, n_embd: usize) -> Result<Self> {
        let embeddings = vs.take("weight", &[vocab_size, n_embd])?;
        Ok(Self { embeddings })
    }

    fn forward(&self, indexes: &XlaOp) -> Result<XlaOp> {
        let embeddings = indexes.builder().constant_literal(&self.embeddings);
        let features = embeddings.take(indexes, 0)?;
        Ok(features)
    }
}

struct LayerNorm {
    scale: Literal,
    bias: Literal,
    size: i64,
}

impl LayerNorm {
    fn new(mut vs: VarStore, size: usize) -> Result<Self> {
        let scale = vs.take("weight", &[size])?;
        let bias = vs.take("bias", &[size])?;
        Ok(Self { scale, bias, size: size as i64 })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let b = x.builder();
        let scale = b.constant_literal(&self.scale).reshape(&[1, 1, self.size])?;
        let bias = b.constant_literal(&self.bias).reshape(&[1, 1, self.size])?;
        let x_norm = x.layer_norm(-1, &scale, &bias)?;
        Ok(x_norm)
    }
}

struct Linear {
    ws: Literal,
    bs: Option<Literal>,
    out_size: usize,
}

impl Linear {
    fn new(mut vs: VarStore, in_size: usize, out_size: usize) -> Result<Self> {
        let ws = vs.take("weight", &[in_size, out_size])?;
        let bs = vs.take("bias", &[out_size])?;
        Ok(Self { ws, bs: Some(bs), out_size })
    }

    fn new_no_bias(mut vs: VarStore, in_size: usize, out_size: usize) -> Result<Self> {
        let ws = vs.take("weight", &[in_size, out_size])?;
        Ok(Self { ws, bs: None, out_size })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let b = x.builder();
        let x_rank = x.rank()?;
        let ws = b.constant_literal(&self.ws);
        let x = x.dot_general(&ws, &[x_rank as i64 - 1], &[0], &[], &[])?;
        let y = match &self.bs {
            None => x,
            Some(bs) => {
                let bs = b.constant_literal(bs).reshape(&[1, 1, self.out_size as i64])?;
                (x + bs)?
            }
        };
        Ok(y)
    }
}

fn masked_fill<T: xla::NativeType>(on_false: &XlaOp, mask: &XlaOp, on_true: T) -> Result<XlaOp> {
    let shape = mask.shape()?;
    let on_true = mask.builder().c0(on_true).broadcast(shape.dimensions())?;
    let m = mask.select(&on_true, on_false)?;
    Ok(m)
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    fn new(vs: VarStore, n_head: usize, n_embd: usize) -> Result<Self> {
        let c_attn = Linear::new(&vs / "c_attn", n_embd, 3 * n_embd)?;
        let c_proj = Linear::new(&vs / "c_proj", n_embd, n_embd)?;
        Ok(Self { c_attn, c_proj, n_head, n_embd })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let (b, t, c) = x.dim3()?;
        let (b, t, c) = (b as i64, t as i64, c as i64);
        let qkv = self.c_attn.forward(x)?;
        let n_embd = self.n_embd as i64;
        let q = qkv.slice_in_dim1(0, n_embd, 2)?;
        let k = qkv.slice_in_dim1(n_embd, 2 * n_embd, 2)?;
        let v = qkv.slice_in_dim1(2 * n_embd, 3 * n_embd, 2)?;
        let target_dim = [b, t, self.n_head as i64, c / self.n_head as i64];
        let k = k.reshape(&target_dim)?.swap_dims(1, 2)?;
        let q = q.reshape(&target_dim)?.swap_dims(1, 2)?;
        let v = v.reshape(&target_dim)?.swap_dims(1, 2)?;
        let k_shape = k.shape()?;
        // TODO: Replace dot_general with matmul.
        let att = (q.dot_general(&k.swap_dims(-2, -1)?, &[3], &[2], &[0, 1], &[0, 1])?
            * builder.c0(1f32 / (k_shape.last_dim().unwrap() as f32).sqrt()))?;
        let mask = builder
            .one(xla::PrimitiveType::S32)
            .broadcast(&[t, t])?
            .lower_triangle()?
            .reshape(&[1, 1, t, t])?;
        let zero =
            builder.zero(xla::PrimitiveType::S32).broadcast(&[b, self.n_head as i64, t, t])?;
        let att = masked_fill(&att, &mask.eq(&zero)?, f32::NEG_INFINITY)?;
        // TODO: Replace dot_general with matmul.
        let y = att.softmax(-1)?.dot_general(&v, &[3], &[2], &[0, 1], &[0, 1])?;
        let y = y.swap_dims(1, 2)?.reshape(&[b, t, c])?;
        let y = self.c_proj.forward(&y)?;
        Ok(y)
    }
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn new(vs: VarStore, config: &GptConfig) -> Result<Self> {
        let c_fc = Linear::new(&vs / "c_fc", config.n_embd, 4 * config.n_embd)?;
        let c_proj = Linear::new(&vs / "c_proj", 4 * config.n_embd, config.n_embd)?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let x = self.c_fc.forward(x)?;
        let x = new_gelu(&x)?;
        self.c_proj.forward(&x)
    }
}

struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: Mlp,
}

struct GptConfig {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self { block_size: 1024, vocab_size: 50257, n_layer: 12, n_head: 12, n_embd: 768 }
    }
}

impl Block {
    fn new(vs: VarStore, config: &GptConfig) -> Result<Self> {
        let ln1 = LayerNorm::new(&vs / "ln_1", config.n_embd)?;
        let attn = CausalSelfAttention::new(&vs / "attn", config.n_head, config.n_embd)?;
        let ln2 = LayerNorm::new(&vs / "ln_2", config.n_embd)?;
        let mlp = Mlp::new(&vs / "mlp", config)?;
        Ok(Self { ln1, attn, ln2, mlp })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let x = (self.attn.forward(&self.ln1.forward(x)?)? + x)?;
        let x = (self.mlp.forward(&self.ln2.forward(&x)?)? + x)?;
        Ok(x)
    }
}

struct Gpt {
    lm_head: Linear,
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
}

impl Gpt {
    fn new(vs: VarStore, config: &GptConfig) -> Result<Self> {
        let lm_head = Linear::new_no_bias(&vs / "lm_head", config.n_embd, config.vocab_size)?;
        let wte = Embedding::new(&vs / "transformer" / "wte", config.vocab_size, config.n_embd)?;
        let wpe = Embedding::new(&vs / "transformer" / "wpe", config.block_size, config.n_embd)?;
        let blocks = (0..config.n_layer)
            .map(|i| Block::new(&vs / "transformer" / "h" / i, config))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = LayerNorm::new(&vs / "transformer" / "ln_f", config.n_embd)?;
        Ok(Self { lm_head, wte, wpe, blocks, ln_f })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let t = x.dim2()?.1 as i64;
        let arange: Vec<_> = (0..t).collect();
        let pos = builder.c1(&arange).reshape(&[1, t])?;

        let tok_emb = self.wte.forward(x)?;
        let pos_emb = self.wpe.forward(&pos)?;
        let mut x = (tok_emb + pos_emb)?;
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.slice_in_dim1(t - 1, t, 1)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }
}

fn gpt_computation(vs: VarStore) -> Result<xla::XlaComputation> {
    let b = XlaBuilder::new("gpt");
    let config = GptConfig::default();
    let gpt = Gpt::new(vs, &config)?;
    let input = b.parameter(0, xla::PrimitiveType::S32, &[2, config.block_size as i64], "tokens");
    let model = gpt.forward(&input)?;
    Ok(model.build()?)
}

fn main() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    println!("{} {} {}", client.platform_name(), client.platform_version(), client.device_count());
    let start_load = std::time::Instant::now();
    let vs = VarStore::new("gpt2.npz")?;
    println!("loaded {} literals in {:?}", vs.len(), start_load.elapsed());
    let start_build = std::time::Instant::now();
    let gpt = gpt_computation(vs)?;
    println!("generated the computation in {:?}", start_build.elapsed());
    let start_compile = std::time::Instant::now();
    let gpt_exe = client.compile(&gpt)?;
    println!("compiled the executable in {:?}", start_compile.elapsed());
    let start_eval = std::time::Instant::now();
    let input = Literal::vec(&vec![42i32; 1024 * 2]).reshape(&[2, 1024])?;
    let result = gpt_exe.execute_literal(&[input])?;
    let result = result[0][0].to_literal_sync()?;
    println!("evaluated the executable in {:?}", start_eval.elapsed());
    println!("{:?}", result.shape());
    let result: Vec<f32> = result.to_vec()?;
    println!("{:?}", &result[..5]);
    Ok(())
}
