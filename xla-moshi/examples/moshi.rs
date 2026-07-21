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
    let mut enc_ctx = xla_moshi::StepCtx::new(&enc_builder, 1 + enc_vb.num_vars() as i64);
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
        let mut inputs: Vec<&xla::PjRtBuffer> = vec![&frame_buf];
        inputs.extend(enc_weights.iter());
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
