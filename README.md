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

Greedy generation of 60 tokens from a 16 token prompt, measured after
compilation and weight loading, on a RTX 4080 SUPER (16GB) for the GPU rows
and a Ryzen 9 7950X (16 cores) for the CPU rows. Prefill is the time to the
first generated token, the decode rate covers the remaining 59 tokens. GPU
numbers are the median of 3 runs, the spread mostly comes from the gemm
autotuner (see the section below). The generated tokens are identical between
the two implementations for all the configurations below, except 0.8B on GPU
bf16 where transformers diverges after 9 tokens because of bf16 rounding (the
f32 outputs of both implementations agree with the xla-rs bf16 tokens).

Decode rate:

|                          | 0.8B        | 2B          | 4B         |
|--------------------------|-------------|-------------|------------|
| GPU bf16, xla-rs         | 341.8 tok/s | 153.7 tok/s | 73.3 tok/s |
| GPU bf16, transformers   | 73.1 tok/s  | 68.8 tok/s  | 51.1 tok/s |
| CPU f32, xla-rs          | 15.7 tok/s  | 6.7 tok/s   | -          |
| CPU f32, transformers    | 7.9 tok/s   | 3.9 tok/s   | -          |

Prefill (time to first token):

|                          | 0.8B   | 2B     | 4B    |
|--------------------------|--------|--------|-------|
| GPU bf16, xla-rs         | 56 ms  | 61 ms  | 77 ms |
| GPU bf16, transformers   | 57 ms  | 58 ms  | 84 ms |
| CPU f32, xla-rs          | 348 ms | 557 ms | -     |
| CPU f32, transformers    | 206 ms | 344 ms | -     |

Note that the xla-rs prefill always processes the full padded 128 token
context (with a sequential scan for the DeltaNet layers), while transformers
only processes the 16 prompt tokens, which explains the slower xla-rs prefill
on cpu. The 4B model in f32 does not fit in the 32GB of RAM of the benchmark
machine.
Versions: xla-rs with xla_extension 0.10.0 (CUDA 13.0 build), transformers
5.14.1 with torch 2.13.0 (cu130 on gpu, cpu wheel on cpu). The transformers
numbers use the plain torch DeltaNet path, the fused flash-linear-attention
kernels are not installed.

### Pinning the gpu autotune configuration

The XLA gemm autotuner benchmarks candidate kernel configurations during
compilation and its choices vary from compile to compile, which moves the gpu
decode rate by up to ~10% on the larger models (63 to 73 tok/s across
compiles of the 4B model). This is the main source of run-to-run variance in
the benchmark above. The autotuner choices can be pinned to a file with the
`--autotune-cache` flag: when the file does not exist the results of the
compilation are written to it, when it exists they are reused, making the
kernel selection deterministic and speeding up the compilation (~14s down to
~4.5s for the 4B model) as all the gemm tuning is skipped:

```bash
cargo run --example qwen35 --release --features hf-hub -- \
  --which 4b --autotune-cache qwen35-4b.pbtxt \
  --prompt "What is the capital of France? Answer in one word."
```

Pinning reproduces whatever run produced the file, good or bad: if the run
that wrote the cache showed a low decode rate, delete the file and rerun
until the tuning run is fast, then keep that file. The cache is keyed by
fusion, gpu, and xla version, so use one file per model size and regenerate
it after changing the model code or the xla_extension build (stale entries
are silently re-tuned, which brings the nondeterminism back). Two smaller
things to be aware of: when loading, the first prefill pays ~130ms of
one-time kernel loading that is otherwise hidden inside the autotuning phase,
and `--xla_gpu_exhaustive_tiling_search` is not a good alternative as it
selects kernels on isolated micro-benchmarks and ends up slower end-to-end.

In the library this is exposed as
`PjRtClient::compile_with_autotune_cache(computation, load_from, dump_to)`
which scopes the `xla_gpu_load_autotune_results_from` and
`xla_gpu_dump_autotune_results_to` debug options to a single compilation (the
same options can also be set process-wide through the `XLA_FLAGS` environment
variable).

## Gemma 4 E2B

The gemma4 example runs the text model of
[Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it), a MatFormer-style
hybrid where four out of five decoder layers use sliding-window attention, the
last twenty layers reuse the k/v states of earlier layers, and a per-layer
embedding is mixed into the residual stream after each mlp. Generation runs in
bf16 with the same prefill/decode split and on-device kv-cache as the qwen35
example, and the `--autotune-cache` flag is supported too.

The repository is gated on the hub: accept the license on the model page and
make a token available via `HF_TOKEN` or `~/.cache/huggingface/token` for the
initial download.

```bash
cargo run --example gemma4 --release --features hf-hub -- \
  --prompt "What is the capital of France? Answer in one word."
```

### Comparison with transformers

Greedy generation of 106 tokens from a 19 token prompt, measured after
compilation, weight loading, and the one-time kernel loading of the first
execution, on the same RTX 4080 SUPER (16GB) and with the same versions as the
qwen35 benchmark above. Prefill is the time to the first generated token
(median of 5), the decode rate covers the remaining 105 tokens (median of 3
runs):

|                          | prefill | decode      |
|--------------------------|---------|-------------|
| GPU bf16, xla-rs         | 12 ms   | 118.0 tok/s |
| GPU bf16, transformers   | 25 ms   | 46.1 tok/s  |

The generated tokens are identical between the two implementations on the
prompts tested. One caveat: on the benchmark prompt the very first token is a
near-tie (the top-2 logits are 0.11 apart in a f32 reference run, about one
bf16 ulp at that magnitude), so bf16-level numeric changes such as a different
autotuned gemm kernel can flip it and the continuations then diverge while
staying equally valid.
