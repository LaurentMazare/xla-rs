// Mimi audio-codec example: encode an audio file to discrete codes and decode
// them back to a waveform, all in a single compiled XLA computation (the
// non-streaming / whole-file path).
//
// The model weights are downloaded automatically from the hugging face hub.
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use xla::{ElementType, PjRtClient, XlaBuilder};
use xla_moshi::mimi::{self, Mimi};
use xla_moshi::Vb;

#[derive(Parser, Debug)]
#[command(name = "moshi", about = "Mimi audio processing tool")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Encode audio to codes and decode back to audio using Mimi.
    AudioToAudio {
        /// Input audio file to process.
        input: std::path::PathBuf,

        /// Output WAV file path.
        #[arg(short, long, default_value = "output.wav")]
        output: std::path::PathBuf,

        /// Number of codebooks to use.
        #[arg(short, long, default_value_t = 16)]
        codebooks: usize,

        /// Use CPU even if an accelerator is available.
        #[arg(long, default_value_t = false)]
        cpu: bool,

        /// Use the streaming per-frame path (compile one static step graph and
        /// run it frame by frame) instead of the whole-file computation.
        #[arg(long, default_value_t = false)]
        streaming: bool,
    },

    /// Run speech-to-text on an audio file (greedy decoding).
    Asr {
        /// Input audio file to process.
        input: std::path::PathBuf,

        /// Use CPU even if an accelerator is available.
        #[arg(long, default_value_t = false)]
        cpu: bool,

        /// Print per-chunk timings instead of the streaming transcript.
        #[arg(long)]
        verbose: bool,

        /// The dtype for the LM weights and computation, f32 or bf16 (the
        /// Mimi audio codec always runs in f32).
        #[arg(long, default_value = "f32")]
        dtype: String,
    },
}

fn download_mimi_model() -> Result<std::path::PathBuf> {
    let repo_id = "kyutai/moshiko-candle-q8";
    let filename = "tokenizer-e351c8d8-checkpoint125.safetensors";
    println!("Downloading mimi model from {repo_id}...");
    let client = hf_hub::HFClientSync::new()?;
    let (owner, name) = repo_id.split_once('/').context("invalid repo")?;
    let path = client.model(owner, name).download_file().filename(filename).send()?;
    println!("  Mimi at {}", path.display());
    Ok(path)
}

fn audio_to_audio(
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    codebooks: usize,
    cpu: bool,
) -> Result<()> {
    let target_sample_rate: usize = 24000;
    let frame_size: usize = 1920;

    // --- Load and resample audio ---
    println!("Loading audio from {}...", input.display());
    let (pcm_data, sample_rate) = kaudio::pcm_decode(&input)?;
    println!(
        "  {} samples at {} Hz ({:.2}s)",
        pcm_data.len(),
        sample_rate,
        pcm_data.len() as f64 / sample_rate as f64
    );
    let pcm_data = if sample_rate as usize != target_sample_rate {
        println!("  Resampling {sample_rate} Hz -> {target_sample_rate} Hz");
        kaudio::resample(&pcm_data, sample_rate as usize, target_sample_rate)?
    } else {
        pcm_data
    };
    let orig_len = pcm_data.len();
    // Pad up to a whole number of frames, as the reference does per chunk.
    let num_frames = orig_len.div_ceil(frame_size);
    let mut pcm_data = pcm_data;
    pcm_data.resize(num_frames * frame_size, 0.0);
    let len = pcm_data.len();
    let audio_duration = orig_len as f64 / target_sample_rate as f64;

    // --- Model + device ---
    let model_path = download_mimi_model()?;
    let client = if cpu { PjRtClient::cpu()? } else { PjRtClient::auto(false)? };
    let config = mimi::Config::v0_1(Some(codebooks));
    println!(
        "  sample_rate={}, frame_rate={}, codebooks={codebooks}",
        config.sample_rate, config.frame_rate
    );

    // --- Build the encode->decode computation ---
    println!("Building the computation ({num_frames} frames)...");
    let start = std::time::Instant::now();
    let builder = XlaBuilder::new("mimi");
    let audio = builder.parameter(0, ElementType::F32, &[1, 1, len as i64], "audio")?;
    let vb_inner = xla_nn::VarBuilder::new(&builder, ElementType::F32, 1);
    let vb = Vb::new(&vb_inner);
    let model = Mimi::load(&vb, config)?;
    let codes = model.encode(&audio)?;
    let decoded = model.decode(&codes)?;
    let computation = builder.tuple(&[&codes, &decoded])?.build()?;
    let exe = client.compile(&computation)?;
    println!("  built and compiled in {:?}", start.elapsed());

    // --- Load weights and run ---
    println!("Loading weights...");
    let start = std::time::Instant::now();
    let weight_buffers = vb_inner.load_buffers(&[&model_path], &client)?;
    // Assert every tensor in the checkpoint was consumed. The codebook running
    // statistics are only used during training, and the checkpoint ships more
    // codebooks than we decode, so those are ignored.
    vb_inner.check_all_used_with_ignore(&[&model_path], |name| {
        name.ends_with("_codebook._initialized")
            || name.ends_with("_codebook.cluster_usage")
            || name.ends_with("_codebook.embedding_sum")
    })?;
    let audio_buffer = client.buffer_from_host_buffer(&pcm_data, &[1, 1, len], None)?;
    let mut inputs: Vec<&xla::PjRtBuffer> = vec![&audio_buffer];
    inputs.extend(weight_buffers.iter());
    println!("  loaded {} weights in {:?}", weight_buffers.len(), start.elapsed());

    println!("Running encode + decode...");
    let start = std::time::Instant::now();
    let outputs = exe.execute_b(&inputs)?;
    let outputs = outputs.into_iter().next().context("no execution outputs")?;
    let codes_lit = outputs[0].to_literal_sync()?;
    let decoded_lit = outputs[1].to_literal_sync()?;
    let elapsed = start.elapsed();

    let codes_shape = codes_lit.array_shape()?;
    let codes_vec = codes_lit.to_vec::<i64>()?;
    println!(
        "  done in {:.2}s ({:.1}x realtime)",
        elapsed.as_secs_f64(),
        audio_duration / elapsed.as_secs_f64()
    );
    println!("Codes shape: {:?} (batch, codebooks, frames)", codes_shape.dims());
    let n_show = codes_vec.len().min(32);
    println!("First {n_show} codes: {:?}", &codes_vec[..n_show]);

    // --- Write the reconstructed audio ---
    let decoded_pcm: Vec<f32> = decoded_lit.to_vec::<f32>()?.into_iter().take(orig_len).collect();
    println!("Writing {} samples to {}...", decoded_pcm.len(), output.display());
    let file = std::fs::File::create(&output)?;
    let mut writer = std::io::BufWriter::new(file);
    kaudio::wav::write_pcm_as_wav(&mut writer, &decoded_pcm, target_sample_rate as u32, 1)?;
    println!("Done.");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::AudioToAudio { input, output, codebooks, cpu, streaming } => {
            if streaming {
                audio_to_audio_streaming(input, output, codebooks, cpu)
            } else {
                audio_to_audio(input, output, codebooks, cpu)
            }
        }
        Command::Asr { input, cpu, verbose, dtype } => run_asr(input, cpu, verbose, &dtype),
    }
}

/// A device buffer of zeros with the given element type and shape, used to
/// initialise streaming state.
fn zeros_buffer(client: &PjRtClient, ty: ElementType, dims: &[i64]) -> Result<xla::PjRtBuffer> {
    let dims_u: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    let n: usize = dims_u.iter().product::<usize>().max(1);
    Ok(match ty {
        ElementType::F32 => client.buffer_from_host_buffer(&vec![0f32; n], &dims_u, None)?,
        ElementType::S32 => client.buffer_from_host_buffer(&vec![0i32; n], &dims_u, None)?,
        ElementType::Bf16 => {
            client.buffer_from_host_raw_bytes(ty, &vec![0u8; 2 * n], &dims_u, None)?
        }
        other => anyhow::bail!("unsupported streaming state dtype {other:?}"),
    })
}

fn audio_to_audio_streaming(
    input: std::path::PathBuf,
    output: std::path::PathBuf,
    codebooks: usize,
    cpu: bool,
) -> Result<()> {
    let target_sample_rate: usize = 24000;
    let frame_size: usize = 1920;

    println!("Loading audio from {}...", input.display());
    let (pcm_data, sample_rate) = kaudio::pcm_decode(&input)?;
    let pcm_data = if sample_rate as usize != target_sample_rate {
        println!("  Resampling {sample_rate} Hz -> {target_sample_rate} Hz");
        kaudio::resample(&pcm_data, sample_rate as usize, target_sample_rate)?
    } else {
        pcm_data
    };
    let orig_len = pcm_data.len();
    let num_frames = orig_len.div_ceil(frame_size);
    let mut pcm_data = pcm_data;
    pcm_data.resize(num_frames * frame_size, 0.0);

    let model_path = download_mimi_model()?;
    let client = if cpu { PjRtClient::cpu()? } else { PjRtClient::auto(false)? };
    let config = mimi::Config::v0_1(Some(codebooks));
    let n_q = codebooks as i64;

    // --- Build the encode-step and decode-step computations (static graphs) ---
    println!("Compiling streaming step graphs...");
    let start = std::time::Instant::now();
    let enc_builder = XlaBuilder::new("mimi-encode-step");
    let frame = enc_builder.parameter(0, ElementType::F32, &[1, 1, frame_size as i64], "frame")?;
    let enc_vb = xla_nn::VarBuilder::new(&enc_builder, ElementType::F32, 1);
    let enc_model = Mimi::load(&Vb::new(&enc_vb), config.clone())?;
    let enc_w = enc_vb.num_vars() as i64;
    // `is_first` (param after the weights) is 1 on the first step and 0 after,
    // so the downsample reproduces its replicate left padding exactly.
    let is_first = enc_builder.parameter(1 + enc_w, ElementType::S32, &[], "is_first")?;
    let mut enc_ctx = xla_moshi::StepCtx::new(&enc_builder, 2 + enc_w);
    enc_ctx.set_is_first(is_first);
    let codes = enc_model.encode_step(&frame, &mut enc_ctx)?;
    let enc_state_shapes: Vec<_> = enc_ctx.state_shapes().to_vec();
    let mut enc_outs = vec![codes];
    enc_outs.extend(enc_ctx.into_new_states());
    let enc_exe =
        client.compile(&enc_builder.tuple(&enc_outs.iter().collect::<Vec<_>>())?.build()?)?;
    let enc_weights = enc_vb.load_buffers(&[&model_path], &client)?;

    let dec_builder = XlaBuilder::new("mimi-decode-step");
    let codes_in = dec_builder.parameter(0, ElementType::S64, &[1, n_q, 1], "codes")?;
    let dec_vb = xla_nn::VarBuilder::new(&dec_builder, ElementType::F32, 1);
    let dec_model = Mimi::load(&Vb::new(&dec_vb), config.clone())?;
    let mut dec_ctx = xla_moshi::StepCtx::new(&dec_builder, 1 + dec_vb.num_vars() as i64);
    let audio_out = dec_model.decode_step(&codes_in, &mut dec_ctx)?;
    let dec_state_shapes: Vec<_> = dec_ctx.state_shapes().to_vec();
    let mut dec_outs = vec![audio_out];
    dec_outs.extend(dec_ctx.into_new_states());
    let dec_exe =
        client.compile(&dec_builder.tuple(&dec_outs.iter().collect::<Vec<_>>())?.build()?)?;
    let dec_weights = dec_vb.load_buffers(&[&model_path], &client)?;
    println!("  compiled in {:?}", start.elapsed());

    // --- Streaming encode: one 1920-sample frame -> one code slice ---
    println!("Streaming encode ({num_frames} frames)...");
    let start = std::time::Instant::now();
    let mut enc_states = enc_state_shapes
        .iter()
        .map(|(ty, d)| zeros_buffer(&client, *ty, d))
        .collect::<Result<Vec<_>>>()?;
    let mut code_slices: Vec<Vec<i64>> = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let chunk = &pcm_data[f * frame_size..(f + 1) * frame_size];
        let frame_buf = client.buffer_from_host_buffer(chunk, &[1, 1, frame_size], None)?;
        let is_first_buf = client.buffer_from_host_buffer(&[i32::from(f == 0)], &[], None)?;
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![&frame_buf];
        inputs.extend(enc_weights.iter());
        inputs.push(&is_first_buf);
        inputs.extend(enc_states.iter());
        let mut out = enc_exe.execute_b(&inputs)?.into_iter().next().context("no enc output")?;
        code_slices.push(out[0].to_literal_sync()?.to_vec::<i64>()?);
        enc_states = out.split_off(1);
    }
    println!("  done in {:?}", start.elapsed());

    // --- Streaming decode: one code slice -> one 1920-sample frame ---
    println!("Streaming decode ({num_frames} frames)...");
    let start = std::time::Instant::now();
    let mut dec_states = dec_state_shapes
        .iter()
        .map(|(ty, d)| zeros_buffer(&client, *ty, d))
        .collect::<Result<Vec<_>>>()?;
    let mut audio: Vec<f32> = Vec::with_capacity(num_frames * frame_size);
    for code_slice in &code_slices {
        let codes_buf = client.buffer_from_host_buffer(code_slice, &[1, n_q as usize, 1], None)?;
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![&codes_buf];
        inputs.extend(dec_weights.iter());
        inputs.extend(dec_states.iter());
        let mut out = dec_exe.execute_b(&inputs)?.into_iter().next().context("no dec output")?;
        audio.extend_from_slice(&out[0].to_literal_sync()?.to_vec::<f32>()?);
        dec_states = out.split_off(1);
    }
    println!("  done in {:?}", start.elapsed());

    let audio: Vec<f32> = audio.into_iter().take(orig_len).collect();
    println!("Writing {} samples to {}...", audio.len(), output.display());
    let file = std::fs::File::create(&output)?;
    let mut writer = std::io::BufWriter::new(file);
    kaudio::wav::write_pcm_as_wav(&mut writer, &audio, target_sample_rate as u32, 1)?;
    println!("Done.");
    Ok(())
}

/// A minimal sentencepiece detokenizer: the piece table is scanned out of the
/// protobuf `.model` file by hand (field 1 of `ModelProto` holds the repeated
/// `SentencePiece` messages, whose field 1 is the piece string and field 3 the
/// piece type). Detokenization concatenates the pieces of the normal tokens,
/// maps the `▁` marker to a space, and drops the leading space, matching
/// `SentencePieceProcessor::Decode`. The `sentencepiece` C++ library is not
/// used on purpose: it links the system protobuf whose symbols clash with the
/// protobuf bundled in `libxla_extension.so` (decoding crashes in a protobuf
/// consistency check when both are loaded).
struct SpDecoder {
    // The piece text for normal tokens, None for control/unknown pieces.
    pieces: Vec<Option<String>>,
}

impl SpDecoder {
    fn open(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mut pieces = vec![];
        let mut pos = 0usize;
        let varint = |data: &[u8], pos: &mut usize| -> Result<u64> {
            let mut v = 0u64;
            let mut shift = 0u32;
            loop {
                let b = *data.get(*pos).context("truncated varint")?;
                *pos += 1;
                v |= u64::from(b & 0x7f) << shift;
                if b & 0x80 == 0 {
                    return Ok(v);
                }
                shift += 7;
            }
        };
        while pos < data.len() {
            let tag = varint(&data, &mut pos)?;
            let (field, wire) = (tag >> 3, tag & 7);
            if field == 1 && wire == 2 {
                // A SentencePiece sub-message.
                let len = varint(&data, &mut pos)? as usize;
                let end = pos + len;
                let (mut piece, mut kind) = (None, 1u64); // type defaults to NORMAL
                while pos < end {
                    let tag = varint(&data, &mut pos)?;
                    match (tag >> 3, tag & 7) {
                        (1, 2) => {
                            let len = varint(&data, &mut pos)? as usize;
                            piece =
                                Some(String::from_utf8_lossy(&data[pos..pos + len]).into_owned());
                            pos += len;
                        }
                        (3, 0) => kind = varint(&data, &mut pos)?,
                        (_, 0) => {
                            varint(&data, &mut pos)?;
                        }
                        (_, 5) => pos += 4,
                        (_, 2) => {
                            let len = varint(&data, &mut pos)? as usize;
                            pos += len;
                        }
                        (f, w) => anyhow::bail!("unsupported piece field {f} wire type {w}"),
                    }
                }
                // 1 = NORMAL, 4 = USER_DEFINED; the rest never produces text.
                pieces.push(piece.filter(|_| kind == 1 || kind == 4));
            } else {
                // Skip the other top-level fields (trainer/normalizer specs).
                match wire {
                    0 => {
                        varint(&data, &mut pos)?;
                    }
                    5 => pos += 4,
                    2 => {
                        let len = varint(&data, &mut pos)? as usize;
                        pos += len;
                    }
                    w => anyhow::bail!("unsupported top-level wire type {w}"),
                }
            }
        }
        Ok(Self { pieces })
    }

    fn decode_piece_ids(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(Some(p)) = self.pieces.get(id as usize) {
                out.push_str(&p.replace('\u{2581}', " "));
            }
        }
        out.strip_prefix(' ').map(str::to_string).unwrap_or(out)
    }
}

/// Download the stt-2.6b-en model files (LM, mimi codec, sentencepiece
/// tokenizer) from the hugging face hub.
fn download_asr_model() -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    let repo_id = "kyutai/stt-2.6b-en-candle";
    println!("Downloading ASR model from {repo_id}...");
    let client = hf_hub::HFClientSync::new()?;
    let (owner, name) = repo_id.split_once('/').context("invalid repo")?;
    let repo = client.model(owner, name);
    let lm = repo.download_file().filename("model.safetensors").send()?;
    let mimi = repo.download_file().filename("mimi-pytorch-e351c8d8@125.safetensors").send()?;
    let tokenizer = repo.download_file().filename("tokenizer_en_audio_4000.model").send()?;
    println!("  LM at {}", lm.display());
    Ok((lm, mimi, tokenizer))
}

// Speech-to-text, mirroring the `asr` command of the `xn-moshi` example: one
// Mimi encode step turns each 80ms frame into a slice of audio codes, one LM
// step turns that slice (plus the previous text token) into the next text
// token, and the text tokens are grouped into words on the host. Text tokens
// 0/2/3/4 (end-of-phrase, end-of-stream, padding, silence padding) delimit
// words, other tokens accumulate into the current word, and the model output
// lags the audio by `ASR_DELAY_SECONDS`.
fn run_asr(input: std::path::PathBuf, cpu: bool, verbose: bool, dtype: &str) -> Result<()> {
    use std::io::Write;

    let lm_dtype = match dtype {
        "f32" => ElementType::F32,
        "bf16" => ElementType::Bf16,
        _ => anyhow::bail!("unsupported dtype {dtype}, expected f32 or bf16"),
    };

    const TARGET_SAMPLE_RATE: usize = 24000;
    const FRAME_SIZE: usize = 1920;
    const ASR_DELAY_SECONDS: f64 = 2.5;
    const WORD_BOUNDARY_TOKENS: [i32; 4] = [0, 2, 3, 4]; // eop, eos, pad, silence pad

    // --- Load and resample audio ---
    println!("Loading audio from {}...", input.display());
    let (pcm_data, sample_rate) = kaudio::pcm_decode(&input)?;
    let audio_duration = pcm_data.len() as f64 / sample_rate as f64;
    println!("  {} samples at {} Hz ({:.2}s)", pcm_data.len(), sample_rate, audio_duration);
    let pcm_data = if sample_rate as usize != TARGET_SAMPLE_RATE {
        println!("  Resampling {sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz");
        kaudio::resample(&pcm_data, sample_rate as usize, TARGET_SAMPLE_RATE)?
    } else {
        pcm_data
    };
    // Two frames of silence before the audio, and the asr delay worth of
    // silence after it so that the lagging transcript flushes completely.
    let pcm_data = [
        vec![0.0; FRAME_SIZE * 2],
        pcm_data,
        vec![0.0; (TARGET_SAMPLE_RATE as f64 * ASR_DELAY_SECONDS) as usize],
    ]
    .concat();

    // --- Model files + tokenizer ---
    let (lm_path, mimi_path, tokenizer_path) = download_asr_model()?;
    let sp = SpDecoder::open(&tokenizer_path)?;
    let client = if cpu { PjRtClient::cpu()? } else { PjRtClient::auto(false)? };

    let lm_config = xla_moshi::lm::Config::stt_2_6b();
    let n_q = lm_config.audio_codebooks;
    let mimi_config = mimi::Config::v0_1(Some(n_q));

    // --- Build the Mimi encode-step computation ---
    println!("Compiling the step graphs...");
    let start = std::time::Instant::now();
    let enc_builder = XlaBuilder::new("mimi-encode-step");
    let frame = enc_builder.parameter(0, ElementType::F32, &[1, 1, FRAME_SIZE as i64], "frame")?;
    let enc_vb = xla_nn::VarBuilder::new(&enc_builder, ElementType::F32, 1);
    let enc_model = Mimi::load(&Vb::new(&enc_vb), mimi_config)?;
    let enc_w = enc_vb.num_vars() as i64;
    let is_first = enc_builder.parameter(1 + enc_w, ElementType::S32, &[], "is_first")?;
    let mut enc_ctx = xla_moshi::StepCtx::new(&enc_builder, 2 + enc_w);
    enc_ctx.set_is_first(is_first);
    let codes = enc_model.encode_step(&frame, &mut enc_ctx)?;
    let enc_state_shapes: Vec<_> = enc_ctx.state_shapes().to_vec();
    let mut enc_outs = vec![codes];
    enc_outs.extend(enc_ctx.into_new_states());
    let enc_exe =
        client.compile(&enc_builder.tuple(&enc_outs.iter().collect::<Vec<_>>())?.build()?)?;

    // --- Build the LM step computation ---
    let lm_builder = XlaBuilder::new("asr-lm-step");
    let codes_in = lm_builder.parameter(0, ElementType::S64, &[1, n_q as i64, 1], "codes")?;
    let text_in = lm_builder.parameter(1, ElementType::S32, &[1], "text_token")?;
    let lm_vb = xla_nn::VarBuilder::new(&lm_builder, lm_dtype, 2);
    let lm_model = xla_moshi::lm::LmModel::load(&Vb::new(&lm_vb), &lm_config)?;
    let mut lm_ctx = xla_moshi::StepCtx::new(&lm_builder, 2 + lm_vb.num_vars() as i64);
    let next_token = lm_model.step(&text_in, &codes_in, &mut lm_ctx)?;
    let lm_state_shapes: Vec<_> = lm_ctx.state_shapes().to_vec();
    let mut lm_outs = vec![next_token];
    lm_outs.extend(lm_ctx.into_new_states());
    let lm_exe =
        client.compile(&lm_builder.tuple(&lm_outs.iter().collect::<Vec<_>>())?.build()?)?;
    println!("  compiled in {:?}", start.elapsed());

    // --- Load the weights ---
    println!("Loading weights...");
    let start = std::time::Instant::now();
    let enc_weights = enc_vb.load_buffers(&[&mimi_path], &client)?;
    enc_vb.check_all_used_with_ignore(&[&mimi_path], |name| {
        name.ends_with("_codebook._initialized")
            || name.ends_with("_codebook.cluster_usage")
            || name.ends_with("_codebook.embedding_sum")
    })?;
    let lm_weights = lm_vb.load_buffers(&[&lm_path], &client)?;
    // The depformer heads (`linears.*`) are only used for audio generation.
    lm_vb.check_all_used_with_ignore(&[&lm_path], |name| name.starts_with("linears."))?;
    println!("  loaded {} weights in {:?}", enc_weights.len() + lm_weights.len(), start.elapsed());

    // --- Streaming ASR loop ---
    let asr_delay_in_tokens =
        (ASR_DELAY_SECONDS * TARGET_SAMPLE_RATE as f64 / FRAME_SIZE as f64) as usize;
    let num_chunks = pcm_data.len().div_ceil(FRAME_SIZE);
    println!("\nProcessing ({num_chunks} chunks of {FRAME_SIZE} samples)...");
    println!("---");
    let start = std::time::Instant::now();

    let mut enc_states = enc_state_shapes
        .iter()
        .map(|(ty, d)| zeros_buffer(&client, *ty, d))
        .collect::<Result<Vec<_>>>()?;
    let mut lm_states = lm_state_shapes
        .iter()
        .map(|(ty, d)| zeros_buffer(&client, *ty, d))
        .collect::<Result<Vec<_>>>()?;
    // The first LM step gets the text start token and padding audio codes (the
    // first slice of real codes is discarded, without shifting the following
    // slices).
    let mut text_buf =
        client.buffer_from_host_buffer(&[lm_model.text_start_token()], &[1], None)?;
    let pad_codes = vec![lm_model.audio_pad_token(); n_q];
    let pad_codes_buf = client.buffer_from_host_buffer(&pad_codes, &[1, n_q, 1], None)?;
    let is_first_true = client.buffer_from_host_buffer(&[1i32], &[], None)?;
    let is_first_false = client.buffer_from_host_buffer(&[0i32], &[], None)?;

    let mut step_idx = 0usize;
    let mut word_tokens: Vec<u32> = vec![];
    // All text tokens with the `3` separator re-inserted between words, so
    // that sentencepiece handles spacing.
    let mut all_text_tokens: Vec<u32> = vec![];
    let mut last_decoded_len = 0;
    // Word assembly: boundary tokens flush the current word, everything else
    // (once past the delay) extends it.
    let mut on_text_token = |text_token: i32| -> Result<()> {
        step_idx += 1;
        if step_idx >= asr_delay_in_tokens {
            if WORD_BOUNDARY_TOKENS.contains(&text_token) {
                if !word_tokens.is_empty() {
                    all_text_tokens.push(3);
                    all_text_tokens.append(&mut word_tokens);
                    let text = sp.decode_piece_ids(&all_text_tokens);
                    if text.len() > last_decoded_len && !verbose {
                        print!("{}", &text[last_decoded_len..]);
                        std::io::stdout().flush()?;
                    }
                    last_decoded_len = text.len();
                }
            } else {
                word_tokens.push(text_token as u32);
            }
        }
        Ok(())
    };

    for f in 0..num_chunks {
        let chunk_start = f * FRAME_SIZE;
        let chunk_end = (chunk_start + FRAME_SIZE).min(pcm_data.len());
        let mut chunk: Vec<f32> = pcm_data[chunk_start..chunk_end].to_vec();
        chunk.resize(FRAME_SIZE, 0.0);
        let step_start = std::time::Instant::now();

        // Mimi encode: one frame of audio -> one slice of codes.
        let frame_buf = client.buffer_from_host_buffer(&chunk, &[1, 1, FRAME_SIZE], None)?;
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![&frame_buf];
        inputs.extend(enc_weights.iter());
        inputs.push(if f == 0 { &is_first_true } else { &is_first_false });
        inputs.extend(enc_states.iter());
        let mut out = enc_exe.execute_b(&inputs)?.into_iter().next().context("no enc output")?;
        let codes_buf = out.remove(0);
        enc_states = out;

        // LM step: the code slice plus the previous text token -> next token.
        let codes_buf = if f == 0 { &pad_codes_buf } else { &codes_buf };
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![codes_buf, &text_buf];
        inputs.extend(lm_weights.iter());
        inputs.extend(lm_states.iter());
        let mut out = lm_exe.execute_b(&inputs)?.into_iter().next().context("no lm output")?;
        let token_buf = out.remove(0);
        lm_states = out;
        // The text token chains on the device (`token_buf` becomes the next
        // step's input), so this step's readback can wait until the next
        // iteration has been dispatched: the host word assembly runs one step
        // behind and the device queue never drains. `text_buf` currently
        // holds the previous step's token, which has not been read yet.
        let prev_token_buf = std::mem::replace(&mut text_buf, token_buf);
        if f > 0 {
            let text_token = prev_token_buf.to_literal_sync()?.to_vec::<i32>()?[0];
            if verbose {
                println!(
                    "  chunk {}/{} -> token {} in {:.2}ms",
                    f + 1,
                    num_chunks,
                    text_token,
                    step_start.elapsed().as_secs_f64() * 1000.0
                );
            }
            on_text_token(text_token)?;
        }
    }
    // Flush the last in-flight token.
    let text_token = text_buf.to_literal_sync()?.to_vec::<i32>()?[0];
    on_text_token(text_token)?;
    println!();
    println!("---");
    if verbose {
        let decoded_text = sp.decode_piece_ids(&all_text_tokens);
        println!("{decoded_text}\n---");
    }
    let elapsed = start.elapsed();
    let audio_duration = pcm_data.len() as f64 / TARGET_SAMPLE_RATE as f64;
    println!(
        "Done in {:.2}s ({:.1}x realtime)",
        elapsed.as_secs_f64(),
        audio_duration / elapsed.as_secs_f64()
    );
    Ok(())
}
