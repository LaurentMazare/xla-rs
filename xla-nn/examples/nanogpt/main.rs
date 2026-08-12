// A very simple GPT implementation based on https://github.com/karpathy/nanoGPT
// This only contains the inference part as the xla crate does not support backpropagation.
// No dropout as this is inference only.
//
// This example requires the following tokenizer config file:
// https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe
// And the gpt2.safetensors weight file that can be extracted by running the get_weights.py script.
use anyhow::Result;
use rand::prelude::*;

extern crate xla;
use xla::{ElementType, PjRtBuffer, PjRtClient, PjRtLoadedExecutable, XlaBuilder, XlaOp};
use xla_nn::{LayerNorm, Linear, Path};

mod tokenizer;
use tokenizer::Tokenizer;

const TY: ElementType = ElementType::F32;
const TEMPERATURE: f32 = 0.8f32;
const USE_CPU: bool = false;
const NUM_SAMPLES: usize = 10;

fn new_gelu(x: &XlaOp) -> Result<XlaOp> {
    let b = x.builder();
    let sqrt_two_over_pi = b.c0((2f32 / std::f32::consts::PI).sqrt())?;
    // 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    let v = (sqrt_two_over_pi * ((b.c0(0.044715f32)? * x.pow(&b.c0(3f32)?)?)? + x)?)?;
    let res = ((b.c0(0.5f32)? * x)? * (v.tanh()? + b.c0(1f32)?)?)?;
    Ok(res)
}

struct Embedding {
    embeddings: XlaOp,
}

impl Embedding {
    fn new(vb: &Path, vocab_size: usize, n_embd: usize) -> Result<Self> {
        let embeddings = vb.var("weight", &[vocab_size as i64, n_embd as i64])?;
        Ok(Self { embeddings })
    }

    fn forward(&self, indexes: &XlaOp) -> Result<XlaOp> {
        let features = self.embeddings.take(indexes, 0)?;
        Ok(features)
    }
}

fn masked_fill<T: xla::NativeType>(on_false: &XlaOp, mask: &XlaOp, on_true: T) -> Result<XlaOp> {
    let shape = mask.array_shape()?;
    let on_true = mask.builder().c0(on_true)?.broadcast(shape.dims())?;
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
    fn new(vb: &Path, n_head: usize, n_embd: usize) -> Result<Self> {
        let d = n_embd as i64;
        let c_attn = Linear::load(&vb.pp("c_attn"), d, 3 * d, true)?;
        let c_proj = Linear::load(&vb.pp("c_proj"), d, d, true)?;
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
        let k_shape = k.array_shape()?;
        let att = (q.matmul(&k.swap_dims(-2, -1)?)?
            * builder.c0(1f32 / (k_shape.last_dim().unwrap() as f32).sqrt()))?;
        let mask = builder
            .one(ElementType::S32)?
            .broadcast(&[t, t])?
            .lower_triangle()?
            .reshape(&[1, 1, t, t])?;
        let zero = builder.zero(ElementType::S32)?.broadcast(&[b, self.n_head as i64, t, t])?;
        let att = masked_fill(&att, &mask.eq(&zero)?, f32::NEG_INFINITY)?;
        let y = att.softmax(-1)?.matmul(&v)?;
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
    fn new(vb: &Path, config: &GptConfig) -> Result<Self> {
        let n_embd = config.n_embd as i64;
        let c_fc = Linear::load(&vb.pp("c_fc"), n_embd, 4 * n_embd, true)?;
        let c_proj = Linear::load(&vb.pp("c_proj"), 4 * n_embd, n_embd, true)?;
        Ok(Self { c_fc, c_proj })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let x = self.c_fc.forward(x)?;
        let x = new_gelu(&x)?;
        Ok(self.c_proj.forward(&x)?)
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
    fn new(vb: &Path, config: &GptConfig) -> Result<Self> {
        let ln1 = LayerNorm::load(&vb.pp("ln_1"), config.n_embd as i64, 1e-5)?;
        let attn = CausalSelfAttention::new(&vb.pp("attn"), config.n_head, config.n_embd)?;
        let ln2 = LayerNorm::load(&vb.pp("ln_2"), config.n_embd as i64, 1e-5)?;
        let mlp = Mlp::new(&vb.pp("mlp"), config)?;
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
    fn new(vb: &Path, config: &GptConfig) -> Result<Self> {
        let lm_head =
            Linear::load(&vb.pp("lm_head"), config.n_embd as i64, config.vocab_size as i64, false)?;
        let transformer = vb.pp("transformer");
        let wte = Embedding::new(&transformer.pp("wte"), config.vocab_size, config.n_embd)?;
        let wpe = Embedding::new(&transformer.pp("wpe"), config.block_size, config.n_embd)?;
        let blocks = (0..config.n_layer)
            .map(|i| Block::new(&transformer.pp("h").pp(i), config))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = LayerNorm::load(&transformer.pp("ln_f"), config.n_embd as i64, 1e-5)?;
        Ok(Self { lm_head, wte, wpe, blocks, ln_f })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let t = x.dim2()?.1 as i64;
        let arange: Vec<_> = (0..t).collect();
        let pos = builder.c1(&arange)?.reshape(&[1, t])?;

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

fn gpt_computation(bsize: i64) -> Result<(xla::XlaComputation, Path)> {
    let b = XlaBuilder::new("gpt");
    // Parameter 0 is the token ids, the weights start at parameter 1.
    let vb = xla_nn::VarBuilder::new(&b, TY, 1).root();
    let config = GptConfig::default();
    let gpt = Gpt::new(&vb, &config)?;
    let input = b.parameter(0, ElementType::S32, &[bsize, config.block_size as i64], "tokens")?;
    let logits = gpt.forward(&input)?;
    let prs = (logits / b.c0(TEMPERATURE))?.softmax(-1)?;
    Ok((prs.build()?, vb))
}

fn sample(
    client: &PjRtClient,
    exe: &PjRtLoadedExecutable,
    weights: &[PjRtBuffer],
    tokenizer: &Tokenizer,
    cnt: usize,
) -> Result<String> {
    let input_str = include_str!("tokenizer.rs");
    let mut input = tokenizer.encode(input_str)?;
    input.pop(); // Remove the <endoftext> token.
    let mut input: Vec<_> = input.into_iter().map(|d| d as i32).collect();
    let mut rng = thread_rng();
    let mut new_tokens = vec![];
    for _i in 1..=cnt {
        let ctxt = &input[input.len().saturating_sub(1024)..];
        let input_b = client.buffer_from_host_buffer(ctxt, &[1, 1024], None)?;
        let mut buffers: Vec<&PjRtBuffer> = vec![&input_b];
        buffers.extend(weights.iter());
        let logits = exe.execute_b(&buffers)?;
        let logits = logits[0][0].to_literal_sync()?;
        let logits_v: Vec<f32> = logits.to_vec()?;
        let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
        let next_token = distr.sample(&mut rng);
        input.push(next_token as i32);
        new_tokens.push(next_token);
    }
    Ok(tokenizer.decode(&new_tokens))
}

fn main() -> Result<()> {
    let client = if USE_CPU { xla::PjRtClient::cpu()? } else { xla::PjRtClient::gpu(0.95, false)? };
    println!("{} {} {}", client.platform_name(), client.platform_version(), client.device_count());
    let tokenizer = Tokenizer::new("vocab.bpe")?;
    println!("loaded tokenizer config, vocab_size: {}", tokenizer.vocab_size());
    let start_build = std::time::Instant::now();
    let (gpt, vb) = gpt_computation(1)?;
    println!("generated the computation in {:?}", start_build.elapsed());
    let start_compile = std::time::Instant::now();
    let gpt_exe = client.compile(&gpt)?;
    println!("compiled the executable in {:?}", start_compile.elapsed());
    let start_load = std::time::Instant::now();
    let weights = vb.var_builder().load_buffers(&["gpt2.safetensors"], &client)?;
    vb.var_builder().check_all_used(&["gpt2.safetensors"])?;
    println!("loaded {} weights in {:?}", weights.len(), start_load.elapsed());
    for _i in 0..NUM_SAMPLES {
        let start_eval = std::time::Instant::now();
        let samples = sample(&client, &gpt_exe, &weights, &tokenizer, 100)?;
        println!("generated the samples in {:?}", start_eval.elapsed());
        println!("----\n{samples}\n----");
    }
    Ok(())
}
