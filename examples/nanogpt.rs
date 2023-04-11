// A very simple GPT implementation based on https://github.com/karpathy/nanoGPT
// This only contains the inference part as the xla crate does not support backpropagation.
// No dropout as this is inference only.
use anyhow::Result;
extern crate xla;
use xla::{Literal, XlaBuilder, XlaOp};

const ET: xla::PrimitiveType = xla::PrimitiveType::F32;

#[allow(dead_code)]
fn new_gelu(x: &XlaOp) -> XlaOp {
    let b = x.builder();
    let sqrt_two_over_pi = b.c0((2f32 / std::f32::consts::PI).sqrt());
    // 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    let v = sqrt_two_over_pi * (b.c0(0.044715f32) * x.pow(&b.c0(3f32)) + x);
    b.c0(0.5f32) * x * (v.tanh() + b.c0(1f32))
}

#[allow(dead_code)]
struct Embedding;

impl Embedding {
    #[allow(dead_code)]
    fn forward(&self, input: &XlaOp) -> XlaOp {
        // TODO
        input.clone()
    }
}

#[allow(dead_code)]
struct LayerNorm {
    scale: XlaOp,
    bias: XlaOp,
}

impl LayerNorm {
    #[allow(dead_code)]
    fn forward(&self, x: &XlaOp) -> XlaOp {
        x.layer_norm(-1, &self.scale, &self.bias)
    }
}

#[allow(dead_code)]
struct Linear {
    ws: XlaOp,
    bs: Option<XlaOp>,
}

impl Linear {
    #[allow(dead_code)]
    fn forward(&self, x: &XlaOp) -> XlaOp {
        let x = x.dot(&self.ws.transpose(&[-2, -1]));
        match &self.bs {
            None => x,
            Some(bs) => x + bs,
        }
    }
}

#[allow(dead_code)]
fn masked_fill<T: xla::NativeType>(on_false: &XlaOp, mask: &XlaOp, on_true: T) -> Result<XlaOp> {
    let shape = mask.shape()?;
    let on_true = mask.builder().c0(on_true).broadcast(shape.dimensions());
    Ok(mask.select(&on_true, on_false))
}

#[allow(dead_code)]
struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    #[allow(dead_code)]
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
        let y = att.softmax(-1).dot(&v);
        let y = y.transpose(&[1, 2]).reshape(x_shape.dimensions());
        let y = self.c_proj.forward(&y);
        Ok(y)
    }
}

#[allow(dead_code)]
struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    #[allow(dead_code)]
    fn forward(&self, x: &XlaOp) -> XlaOp {
        let x = self.c_fc.forward(x);
        let x = new_gelu(&x);
        self.c_proj.forward(&x)
    }
}

#[allow(dead_code)]
struct Block {
    ln1: LayerNorm,
    attn: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    #[allow(dead_code)]
    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let x = self.attn.forward(&self.ln1.forward(x))? + x;
        let x = self.mlp.forward(&self.ln2.forward(&x)) + x;
        Ok(x)
    }
}

#[allow(dead_code)]
struct Gpt {
    lm_head: Linear,
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
}

impl Gpt {
    #[allow(dead_code)]
    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let x_shape = x.shape()?;
        let (_b, t) = <(i64, i64)>::try_from(&x_shape)?;
        let arange: Vec<_> = (0..t).collect();
        let pos = builder.c1(&arange).convert_element_type(ET);

        let tok_emb = self.wte.forward(x);
        let pos_emb = self.wpe.forward(&pos);
        let mut x = tok_emb + pos_emb;
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        let x = self.ln_f.forward(&x);
        let logits = self.lm_head.forward(&x.slice_in_dim(-1, -1, 1, 1));
        Ok(logits)
    }
}

fn gpt_computation() -> xla::Result<xla::XlaComputation> {
    let b = XlaBuilder::new("gpt");
    let model = b.constant_r0(42f32);
    model.build()
}

fn main() -> Result<()> {
    let client = xla::PjRtClient::cpu()?;
    println!("{} {} {}", client.platform_name(), client.platform_version(), client.device_count());
    let gpt = gpt_computation()?;
    let gpt_exe = client.compile(&gpt)?;
    let _result = gpt_exe.execute_literal(&[Literal::from(12f32)])?;
    Ok(())
}
