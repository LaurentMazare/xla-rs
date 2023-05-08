// An implementation of LLaMA https://github.com/facebookresearch/llama
// This only contains the inference part as the xla crate does not support backpropagation.
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/blob/main/tokenizer.json
//
// In order to convert the llama weights to a .npz file, run:
// python examples/llama/convert_checkpoint.py ..../LLaMA/7B/consolidated.00.pth
use anyhow::Result;
use clap::Parser;
use rand::prelude::*;

extern crate xla;
use xla::{ElementType, PrimitiveType, XlaBuilder, XlaOp};

mod sentencepiece;
use sentencepiece::Tokenizer;
mod var_store;
use var_store::{VarBuilder, VarStore};

const CONTEXT_SIZE: usize = 512;
const START_PROMPT: &str = r"
EDWARD:
I wonder how our princely father 'scaped,
Or whether he be 'scaped away or no
From Clifford's and Northumberland's pursuit:
Had he been ta'en, we should have heard the news;
Had he been slain, we should have heard the news;
Or had he 'scaped, methinks we should have heard
The happy tidings of his good escape.
How fares my brother? why is he so sad?

RICHARD:
I cannot joy, until I be resolved
Where our right valiant father is become.
I saw him in the battle range about;
And watch'd him how he singled Clifford forth.
Methought he bore him in the thickest troop
As doth a lion in a herd of neat;
Or as a bear, encompass'd round with dogs,
Who having pinch'd a few and made them cry,
The rest stand all aloof, and bark at him.
So fared our father with his enemies;
So fled his enemies my warlike father:
Methinks, 'tis prize enough to be his son.
See how the morning opes her golden gates,
And takes her farewell of the glorious sun!
How well resembles it the prime of youth,
Trimm'd like a younker prancing to his love!

EDWARD:
Dazzle mine eyes, or do I see three suns?

RICHARD:
Three glorious suns, each one a perfect sun;
Not separated with the racking clouds,
But sever'd in a pale clear-shining sky.
See, see! they join, embrace, and seem to kiss,
As if they vow'd some league inviolable:
Now are they but one lamp, one light, one sun.
In this the heaven figures some event.

EDWARD:
'Tis wondrous strange, the like yet never heard of.
I think it cites us, brother, to the field,
That we, the sons of brave Plantagenet,
Each one already blazing by our meeds,
Should notwithstanding join our lights together
And over-shine the earth as this the world.
Whate'er it bodes, henceforward will I bear
Upon my target three fair-shining suns.
";

#[allow(dead_code)]
struct Config {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
}

#[allow(dead_code)]
impl Config {
    fn config_7b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 32, n_head: 32, n_embd: 4096 }
    }

    fn config_13b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 40, n_head: 40, n_embd: 5120 }
    }

    fn config_30b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 60, n_head: 52, n_embd: 6656 }
    }

    fn config_65b() -> Self {
        Self { block_size: 4096, vocab_size: 32000, n_layer: 80, n_head: 64, n_embd: 8192 }
    }
}

struct Embedding {
    embeddings: XlaOp,
}

impl Embedding {
    fn new(mut vb: VarBuilder, vocab_size: usize, n_embd: usize) -> Result<Self> {
        let embeddings = vb.var("weight", &[vocab_size, n_embd])?;
        Ok(Self { embeddings })
    }

    fn forward(&self, indexes: &XlaOp) -> Result<XlaOp> {
        let features = self.embeddings.take(indexes, 0)?;
        Ok(features)
    }
}

struct Linear {
    ws: XlaOp,
    bs: Option<XlaOp>,
    out_size: usize,
}

impl Linear {
    #[allow(dead_code)]
    fn new(mut vb: VarBuilder, in_size: usize, out_size: usize) -> Result<Self> {
        let ws = vb.var("weight", &[in_size, out_size])?;
        let bs = vb.var("bias", &[out_size])?;
        Ok(Self { ws, bs: Some(bs), out_size })
    }

    fn new_no_bias(mut vb: VarBuilder, in_size: usize, out_size: usize) -> Result<Self> {
        let ws = vb.var("weight", &[in_size, out_size])?;
        Ok(Self { ws, bs: None, out_size })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let x_rank = x.rank()?;
        let x = x.dot_general(&self.ws, &[x_rank as i64 - 1], &[0], &[], &[])?;
        let y = match &self.bs {
            None => x,
            Some(bs) => {
                let bs = bs.reshape(&[1, 1, self.out_size as i64])?;
                (x + bs)?
            }
        };
        Ok(y)
    }
}

struct RmsNorm {
    scale: XlaOp,
    size: i64,
}

impl RmsNorm {
    fn new(mut vb: VarBuilder, size: usize) -> Result<Self> {
        let scale = vb.var("scale", &[size])?;
        Ok(Self { scale, size: size as i64 })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let eps = builder.c0(1e-5)?.convert(x.ty()?)?;
        let norm_x = (x * x)?.reduce_mean(&[-1], true)?;
        let x_normed = (x * (norm_x + eps)?.rsqrt()?)?;
        let scale = self.scale.reshape(&[1, 1, self.size])?;
        Ok((scale * x_normed)?)
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder, n_embd: usize) -> Result<Self> {
        let n_hidden = 8 * n_embd / 3;
        let n_hidden = (n_hidden - 1) / 256 * 256 + 256;
        let c_fc1 = Linear::new_no_bias(&vb / "c_fc1", n_embd, n_hidden)?;
        let c_fc2 = Linear::new_no_bias(&vb / "c_fc2", n_embd, n_hidden)?;
        let c_proj = Linear::new_no_bias(&vb / "c_proj", n_hidden, n_embd)?;
        Ok(Self { c_fc1, c_fc2, c_proj })
    }

    fn forward(&self, x: &XlaOp) -> Result<XlaOp> {
        let x = (self.c_fc1.forward(x)?.silu()? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x)
    }
}

fn masked_fill<T: xla::NativeType>(on_false: &XlaOp, mask: &XlaOp, on_true: T) -> Result<XlaOp> {
    let shape = mask.array_shape()?;
    let on_true = mask.builder().c0(on_true)?.convert(on_false.ty()?)?.broadcast(shape.dims())?;
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
    fn new(vb: VarBuilder, n_head: usize, n_embd: usize) -> Result<Self> {
        let c_attn = Linear::new_no_bias(&vb / "c_attn", n_embd, 3 * n_embd)?;
        let c_proj = Linear::new_no_bias(&vb / "c_proj", n_embd, n_embd)?;
        Ok(Self { c_attn, c_proj, n_head, n_embd })
    }

    fn apply_rotary_emb(&self, x: &XlaOp, freqs_cis: &XlaOp) -> Result<XlaOp> {
        let mut dims: Vec<_> = x.dims()?.into_iter().map(|c| c as i64).collect();
        let v = dims.pop().unwrap();
        dims.push(v / 2);
        dims.push(2);
        let x = x.reshape(&dims)?;
        let re_x = x.slice_in_dim1(0, 1, -1)?;
        let im_x = x.slice_in_dim1(1, 2, -1)?;
        let re_f = freqs_cis.slice_in_dim1(0, 1, -1)?;
        let im_f = freqs_cis.slice_in_dim1(1, 2, -1)?;
        let re = ((&re_x * &re_f)? - (&im_x * &im_f)?)?;
        let im = ((&re_x * &im_f)? + (&im_x * &re_f)?)?;
        let rope = re.concat_in_dim(&[&im], -1)?;
        // TODO: Add the flatten op.
        let mut dims: Vec<_> = rope.dims()?.into_iter().map(|c| c as i64).collect();
        let v1 = dims.pop().unwrap();
        let v2 = dims.pop().unwrap();
        dims.push(v1 * v2);
        let rope = rope.reshape(&dims)?;
        Ok(rope)
    }

    fn forward(&self, x: &XlaOp, freqs_cis: &XlaOp) -> Result<XlaOp> {
        let builder = x.builder();
        let ty = x.ty()?;
        let freqs_cis = freqs_cis.convert(ty)?;
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
        let q = self.apply_rotary_emb(&q, &freqs_cis)?;
        let k = self.apply_rotary_emb(&k, &freqs_cis)?;
        let k_shape = k.array_shape()?;
        let att = (q.matmul(&k.swap_dims(-2, -1)?)?
            * builder.c0(1f32 / (k_shape.last_dim().unwrap() as f32).sqrt())?.convert(ty)?)?;
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

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let rms_1 = RmsNorm::new(&vb / "rms_1", config.n_embd)?;
        let attn = CausalSelfAttention::new(&vb / "attn", config.n_head, config.n_embd)?;
        let rms_2 = RmsNorm::new(&vb / "rms_2", config.n_embd)?;
        let mlp = Mlp::new(&vb / "mlp", config.n_embd)?;
        Ok(Self { rms_1, attn, rms_2, mlp })
    }

    fn forward(&self, x: &XlaOp, freqs_cis: &XlaOp) -> Result<XlaOp> {
        let x = (self.attn.forward(&self.rms_1.forward(x)?, freqs_cis)? + x)?;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + x)?;
        Ok(x)
    }
}

struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let lm_head = Linear::new_no_bias(&vb / "lm_head", config.n_embd, config.vocab_size)?;
        let wte = Embedding::new(&vb / "transformer" / "wte", config.vocab_size, config.n_embd)?;
        let blocks = (0..config.n_layer)
            .map(|i| Block::new(&vb / "transformer" / "h" / i, config))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = RmsNorm::new(&vb / "transformer" / "ln_f", config.n_embd)?;
        Ok(Self { wte, blocks, ln_f, lm_head })
    }

    fn forward(&self, x: &XlaOp, freqs_cis: &XlaOp) -> Result<XlaOp> {
        let t = x.dim2()?.1 as i64;
        let mut x = self.wte.forward(x)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, freqs_cis)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.slice_in_dim1(t - 1, t, 1)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }
}

fn precompute_freqs_cis(config: &Config, builder: &XlaBuilder) -> Result<XlaOp> {
    let seq_len = CONTEXT_SIZE;
    let n_elem = config.n_embd / config.n_head;
    let theta: Vec<_> =
        (0..n_elem).step_by(2).map(|i| 1f32 / 10000f32.powf(i as f32 / n_elem as f32)).collect();
    let arange: Vec<_> = (0..seq_len).map(|c| c as f32).collect();
    let theta = builder.c1::<f32>(&theta)?;
    let arange = builder.c1::<f32>(&arange)?;
    let idx_theta = arange.dot_general(&theta, &[], &[], &[], &[])?;
    let shape = [1, 1, seq_len as i64, n_elem as i64 / 2, 1];
    let idx_theta_cos = idx_theta.cos()?.reshape(&shape)?;
    let idx_theta_sin = idx_theta.sin()?.reshape(&shape)?;
    Ok(idx_theta_cos.concat_in_dim(&[&idx_theta_sin], -1)?)
}

fn llama_computation(args: &Args, bsize: i64) -> Result<(xla::XlaComputation, VarStore)> {
    let b = XlaBuilder::new("llama");
    let mut vb = if args.cpu {
        VarBuilder::new::<xla::F16, f32>(&b)
    } else {
        VarBuilder::new::<xla::F16, xla::Bf16>(&b)
    };
    let config = Config::config_7b();
    let freqs_cis = precompute_freqs_cis(&config, &b)?;
    let llama = Llama::new(vb.clone(), &config)?;
    let input = vb.arg("tokens", ElementType::U32, &[bsize as usize, CONTEXT_SIZE])?;
    let logits = llama.forward(&input, &freqs_cis)?.convert(PrimitiveType::F32)?;
    let prs = (logits / b.c0(args.temperature)?)?.softmax(-1)?;
    Ok((prs.build()?, vb.into_store()))
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizer = Tokenizer::from_file("llama-tokenizer.json")?;
    let mut tokens = tokenizer.encode(START_PROMPT)?;
    let mut new_tokens = vec![];
    let client =
        if args.cpu { xla::PjRtClient::cpu()? } else { xla::PjRtClient::gpu(0.95, false)? };
    println!("{} {} {}", client.platform_name(), client.platform_version(), client.device_count());
    let start_build = std::time::Instant::now();
    let (llama, mut vs) = llama_computation(&args, 1)?;
    println!("generated the computation in {:?}", start_build.elapsed());
    let start_compile = std::time::Instant::now();
    let llama_exe = client.compile(&llama)?;
    println!("compiled the executable in {:?}", start_compile.elapsed());
    let start_load = std::time::Instant::now();
    let mut buffers = vs.load_from_npz("llama.npz", &client)?;
    let arg_index = vs.arg_indexes()[0];
    println!("loaded weights in {:?} ({arg_index})", start_load.elapsed());
    let mut rng = thread_rng();
    for index in 0..args.sample_len {
        let ctxt: Vec<_> =
            tokens[tokens.len().saturating_sub(CONTEXT_SIZE)..].iter().map(|c| *c as u32).collect();
        buffers[arg_index] = client.buffer_from_host_buffer(&ctxt, &[1, CONTEXT_SIZE], None)?;
        let logits = llama_exe.execute_b(&buffers)?;
        let logits = logits[0][0].to_literal_sync()?;
        let logits_v: Vec<f32> = logits.to_vec()?;
        let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
        let next_token = distr.sample(&mut rng);
        tokens.push(next_token);
        new_tokens.push(next_token);
        println!("{} token: {} '{}'", index + 1, next_token, tokenizer.decode(&[next_token]));
    }
    println!("----\n{}\n----", tokenizer.decode(&new_tokens));
    Ok(())
}
