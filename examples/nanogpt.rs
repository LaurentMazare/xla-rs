// A very simple GPT implementation based on https://github.com/karpathy/nanoGPT
// This only contains the inference part as the xla crate does not support backpropagation.
// No dropout as this is inference only.
use anyhow::Result;
extern crate xla;
use xla::{Literal, XlaBuilder, XlaOp};

const ET: xla::PrimitiveType = xla::PrimitiveType::F32;

fn new_gelu(x: &XlaOp) -> XlaOp {
    let b = x.builder();
    let sqrt_two_over_pi = b.c0((2f32 / std::f32::consts::PI).sqrt());
    // 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    let v = sqrt_two_over_pi * (b.c0(0.044715f32) * x.pow(&b.c0(3f32)) + x);
    b.c0(0.5f32) * x * (v.tanh() + b.c0(1f32))
}

struct Embedding {
    embeddings: Literal,
}

impl Embedding {
    fn new(vocab_size: usize, n_embd: usize) -> Self {
        // TODO
        let embeddings = Literal::create_from_shape(ET, &[vocab_size, n_embd]);
        Self { embeddings }
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
}

impl LayerNorm {
    fn new(size: usize) -> Self {
        // TODO
        let scale = Literal::create_from_shape(ET, &[size]);
        let bias = Literal::create_from_shape(ET, &[size]);
        Self { scale, bias }
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let b = x.builder();
        let scale = b.constant_literal(&self.scale);
        let bias = b.constant_literal(&self.bias);
        let x_norm = x.layer_norm(-1, &scale, &bias)?;
        Ok(x_norm)
    }
}

struct Linear {
    ws: Literal,
    bs: Option<Literal>,
}

impl Linear {
    fn new(in_size: usize, out_size: usize) -> Self {
        // TODO
        let ws = Literal::create_from_shape(ET, &[in_size, out_size]);
        let bs = Literal::create_from_shape(ET, &[out_size]);
        Self { ws, bs: Some(bs) }
    }

    fn forward(&self, x: &XlaOp) -> XlaOp {
        let b = x.builder();
        let ws = b.constant_literal(&self.ws);
        let x = x.dot(&ws.transpose(&[-2, -1]));
        match &self.bs {
            None => x,
            Some(bs) => {
                let bs = b.constant_literal(bs);
                x + bs
            }
        }
    }
}

fn masked_fill<T: xla::NativeType>(on_false: &XlaOp, mask: &XlaOp, on_true: T) -> Result<XlaOp> {
    let shape = mask.shape()?;
    let on_true = mask.builder().c0(on_true).broadcast(shape.dimensions());
    Ok(mask.select(&on_true, on_false))
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    fn new(n_head: usize, n_embd: usize) -> Self {
        let c_attn = Linear::new(n_embd, 3 * n_embd);
        let c_proj = Linear::new(n_embd, n_embd);
        Self { c_attn, c_proj, n_head, n_embd }
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let x_shape = x.shape()?;
        let (b, t, c) = <(i64, i64, i64)>::try_from(&x_shape)?;
        let qkv = self.c_attn.forward(x);
        let n_embd = self.n_embd as i64;
        let q = qkv.slice_in_dim(0, n_embd, 1, 2);
        let k = qkv.slice_in_dim(n_embd, 2 * n_embd, 1, 2);
        let v = qkv.slice_in_dim(2 * n_embd, 3 * n_embd, 1, 2);
        let target_dim = [b, t, c / self.n_head as i64];
        let k = k.reshape(&target_dim).transpose(&[1, 2]);
        let q = q.reshape(&target_dim).transpose(&[1, 2]);
        let v = v.reshape(&target_dim).transpose(&[1, 2]);
        let k_shape = k.shape()?;
        let att = q.dot(&k.transpose(&[-2, -1]))
            * builder.c0(1f32 / (k_shape.last_dim().unwrap() as f32).sqrt());
        let mask = builder.one(ET).broadcast(&[t, t]).lower_triangle().broadcast(&[1, 1, t, t]);
        let att = masked_fill(&att, &mask.eq(&builder.c0(0f32)), f32::NEG_INFINITY)?;
        let y = att.softmax(-1)?.dot(&v);
        let y = y.transpose(&[1, 2]).reshape(x_shape.dimensions());
        let y = self.c_proj.forward(&y);
        Ok(y)
    }
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn new(config: &GptConfig) -> Self {
        let c_fc = Linear::new(config.n_embd, 4 * config.n_embd);
        let c_proj = Linear::new(4 * config.n_embd, config.n_embd);
        Self { c_fc, c_proj }
    }

    fn forward(&self, x: &XlaOp) -> XlaOp {
        let x = self.c_fc.forward(x);
        let x = new_gelu(&x);
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
        Self { block_size: 1024, vocab_size: 50304, n_layer: 12, n_head: 12, n_embd: 768 }
    }
}

impl Block {
    fn new(config: &GptConfig) -> Self {
        let ln1 = LayerNorm::new(config.n_embd);
        let attn = CausalSelfAttention::new(config.n_head, config.n_embd);
        let ln2 = LayerNorm::new(config.n_embd);
        let mlp = Mlp::new(config);
        Self { ln1, attn, ln2, mlp }
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let x = self.attn.forward(&self.ln1.forward(x)?)? + x;
        let x = self.mlp.forward(&self.ln2.forward(&x)?) + x;
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

fn debug(builder: &XlaBuilder, message: &str) -> Result<()> {
    let status = builder.get_current_status();
    println!("{message} {status:?}");
    status?;
    Ok(())
}

impl Gpt {
    fn new(config: &GptConfig) -> Result<Self> {
        let lm_head = Linear::new(config.n_embd, config.vocab_size);
        let wte = Embedding::new(config.vocab_size, config.n_embd);
        let wpe = Embedding::new(config.block_size, config.n_embd);
        let blocks = (0..config.n_layer).map(|_i| Block::new(config)).collect();
        let ln_f = LayerNorm::new(config.n_embd);
        Ok(Self { lm_head, wte, wpe, blocks, ln_f })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let x_shape = x.shape()?;
        let (_b, t) = <(i64, i64)>::try_from(&x_shape)?;
        let arange: Vec<_> = (0..t).collect();
        let pos = builder.c1(&arange).reshape(&[1, t]);

        let tok_emb = self.wte.forward(x)?;
        let pos_emb = self.wpe.forward(&pos)?;
        debug(builder, "post embedding")?;
        let mut x = tok_emb + pos_emb;
        debug(builder, "pre blocks")?;
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        debug(builder, "post blocks")?;
        let x = self.ln_f.forward(&x)?;
        debug(builder, "post ln_f")?;
        let logits = self.lm_head.forward(&x.slice_in_dim(-1, -1, 1, 1));
        Ok(logits)
    }
}

fn gpt_computation() -> Result<xla::XlaComputation> {
    let b = XlaBuilder::new("gpt");
    let config = GptConfig::default();
    let gpt = Gpt::new(&config)?;
    let input = b.parameter(0, xla::PrimitiveType::S32, &[2, config.block_size as i64], "tokens");
    let model = gpt.forward(&input)?;
    Ok(model.build()?)
}

fn main() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    println!("{} {} {}", client.platform_name(), client.platform_version(), client.device_count());
    let gpt = gpt_computation()?;
    let gpt_exe = client.compile(&gpt)?;
    let _result = gpt_exe.execute_literal(&[Literal::from(12f32)])?;
    Ok(())
}
