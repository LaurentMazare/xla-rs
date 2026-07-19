# xla-rs
Experimentation using the xla compiler from rust

Pre-compiled binaries for the xla library can be downloaded from the
[elixir-nx/xla repo](https://github.com/elixir-nx/xla/releases/tag/v0.10.0).
These should be extracted at the root of this repository, resulting
in a `xla_extension` subdirectory being created, the currently supported version
is 0.10.0.

For a linux platform, this can be done via:
```bash
wget https://github.com/elixir-nx/xla/releases/download/v0.10.0/xla_extension-0.10.0-x86_64-linux-gnu-cpu.tar.gz
tar -xzvf xla_extension-0.10.0-x86_64-linux-gnu-cpu.tar.gz
```

If the `xla_extension` directory is not in the main project directory, the path
can be specified via the `XLA_EXTENSION_DIR` environment variable.

## Generating some Text Samples with Qwen3.5

The [Qwen3.5 models](https://huggingface.co/Qwen/Qwen3.5-2B) can be used to
generate text. They use a hybrid architecture mixing gated DeltaNet linear
attention layers with full attention layers. The example loads the safetensors
weights directly and runs greedy generation with a kv-cache and DeltaNet state
carry-over.

The weights and tokenizer config are downloaded automatically from the hub
(and cached locally). Generation runs in bf16 by default, `--dtype f32` and
`--dtype f16` are also supported. The model size is selected with `--which`,
one of `0.8b`, `2b` (the default), `4b`, or `9b`:

```bash
# Run the example, use --cpu to run on cpu rather than gpu.
cargo run --example qwen35 --release --features hf-hub -- \
  --which 2b \
  --prompt "What is the capital of France? Answer in one word." \
  --sample-len 30
```

### Comparison with transformers

Greedy generation of 60 tokens, measured after compilation and weight loading,
on a RTX 4080 SUPER (16GB) for the GPU rows and a Ryzen 9 7950X (16 cores) for
the CPU rows. The generated tokens are identical between the two
implementations for all the configurations below, except 0.8B on GPU bf16
where transformers diverges after 9 tokens because of bf16 rounding (the f32
outputs of both implementations agree with the xla-rs bf16 tokens).

|                          | 0.8B        | 2B         | 4B         |
|--------------------------|-------------|------------|------------|
| GPU bf16, xla-rs         | 158.0 tok/s | 74.8 tok/s | 41.7 tok/s |
| GPU bf16, transformers   | 69.9 tok/s  | 65.4 tok/s | 46.8 tok/s |
| CPU f32, xla-rs          | 14.7 tok/s  | 6.5 tok/s  | -          |
| CPU f32, transformers    | 7.6 tok/s   | 3.7 tok/s  | -          |

The 4B model in f32 does not fit in the 32GB of RAM of the benchmark machine.
Versions: xla-rs with xla_extension 0.10.0 (CUDA 13.0 build), transformers
5.14.1 with torch 2.13.0 (cu130 on gpu, cpu wheel on cpu). The transformers
numbers use the plain torch DeltaNet path, the fused flash-linear-attention
kernels are not installed.
