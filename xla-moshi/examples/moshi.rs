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
        Command::AudioToAudio { input, output, codebooks, cpu } => {
            audio_to_audio(input, output, codebooks, cpu)
        }
    }
}
