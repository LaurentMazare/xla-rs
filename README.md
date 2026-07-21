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

## Gemma 4 E2B and E4B

The gemma4 example runs the text model of
[Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) or
[E4B](https://huggingface.co/google/gemma-4-E4B-it) (`--which e2b|e4b`),
MatFormer-style hybrids where most decoder layers use sliding-window
attention, the last layers reuse the k/v states of earlier layers, and a
per-layer embedding (PLE) is mixed into the residual stream after each mlp.
Generation runs in bf16 with the same prefill/decode split and on-device
kv-cache as the qwen35 example, and the `--autotune-cache` flag is supported
too.

The PLE table is the "effective" in the E model names: it holds roughly half
of the parameters (2.4B of E2B's 4.6B, 2.9B of E4B's 7.5B) but is only ever
read one row per token, so by default the example keeps it in host memory
and gathers the rows for the current tokens on the cpu, passing them to the
computations as an input. This halves the device memory (4.5GB of weights
for E2B, 9.2GB for E4B) and is what lets E4B run on a 16GB gpu in bf16, at
the cost of a host round-trip per decode step: the next step can only be
launched once the generated token is back on the host, while with the table
on the device the token is chained on the device as in the qwen35 example.
The `--ple-on-device` flag keeps the table in device memory instead (an
extra 4.8GB for E2B, and 5.8GB for E4B which then no longer fits on a 16GB
gpu), trading the memory back for the chained decode loop.

The repositories are gated on the hub: accept the license on the model page
and make a token available via `HF_TOKEN` or `~/.cache/huggingface/token` for
the initial download.

```bash
cargo run --example gemma4 --release --features hf-hub -- \
  --which e4b \
  --prompt "What is the capital of France? Answer in one word."
```

### Comparison with transformers

Greedy generation of 106 tokens from a 19 token prompt, measured after
compilation, weight loading, and the one-time kernel loading of the first
execution, on the same RTX 4080 SUPER (16GB) and with the same versions as the
qwen35 benchmark above. Prefill is the time to the first generated token
(median of 5), the decode rate covers the remaining 105 tokens (median of 3
runs), and the memory column is the peak gpu usage of the xla-rs run:

|                          | E2B         | E2B, PLE on device | E4B        |
|--------------------------|-------------|--------------------|------------|
| prefill, xla-rs          | 13 ms       | 12 ms              | 24 ms      |
| prefill, transformers    | 25 ms       | 25 ms              | -          |
| decode, xla-rs           | 110.1 tok/s | 117.4 tok/s        | 48.0 tok/s |
| decode, transformers     | 46.1 tok/s  | 46.1 tok/s         | -          |
| gpu memory, xla-rs       | 8.5 GB      | 9.6 GB             | 14.7 GB    |

The host round-trip of the default PLE offload costs E2B ~6% decode rate
(110 vs 117 tok/s with `--ple-on-device`) for 1.1GB less peak gpu memory
(the 4.8GB table plus its gathers is mostly hidden by the XLA workspace in
the peak numbers). The transformers E4B column is empty as the full bf16
text model (15GB of weights with the PLE table on the device) does not leave
enough room on a 16GB gpu; splitting it across gpu and cpu with accelerate
works for checking outputs but is not a meaningful speed comparison. The
generated tokens are identical between the two implementations on the
prompts tested (E4B was checked against a gpu+cpu split run). One caveat: on
E2B the very first token of the benchmark prompt is a near-tie (the top-2
logits are 0.11 apart in a f32 reference run, about one bf16 ulp at that
magnitude), so bf16-level numeric changes such as a different autotuned gemm
kernel or the PLE offload's slightly different graph can flip it and the
continuations then diverge while staying equally valid.

## Nemotron 3 Nano 4B

The nemotron3 example runs
[NVIDIA Nemotron 3 Nano 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16),
a Nemotron-H style hybrid where each of the 42 layers holds a single mixer:
a Mamba2 SSM (21 layers), a plain squared-relu MLP (17 layers), or full
attention with GQA (4 layers). The attention layers use no positional
embedding: positional information only comes from the recurrent Mamba2
layers. As in the qwen35 example the prefill computation runs the recurrence
sequentially with an XLA while loop, and the decode computation carries the
SSM state (kept in f32), the conv1d window, and the k/v caches on the device
across steps. Generation runs in bf16 and the `--autotune-cache` flag is
supported.

```bash
cargo run --example nemotron3 --release --features hf-hub -- \
  --prompt "What is the capital of France? Answer in one word."
```

### Comparison with transformers

Greedy generation of 100 tokens from a 28 token prompt, measured after
compilation, weight loading, and the one-time kernel loading of the first
execution, on the same RTX 4080 SUPER (16GB) and with the same versions as
the qwen35 benchmark above. Prefill is the time to the first generated token
(median of 5), the decode rate covers the remaining 99 tokens (median of 3
runs), and the memory column is the peak gpu usage of the xla-rs run:

|                          | prefill  | decode     | gpu memory |
|--------------------------|----------|------------|------------|
| GPU bf16, xla-rs         | 24 ms    | 89.1 tok/s | 8.5 GB     |
| GPU bf16, transformers   | 483 ms   | 71.7 tok/s | -          |

The transformers numbers use the naive torch path for the Mamba2 layers as
the fused `mamba-ssm`/`causal-conv1d` kernels are not installed; most of its
prefill gap comes from that path materializing the full chunked SSD
intermediates. The generated tokens are identical between the two
implementations on the prompts tested, with one exception on a 100 token run
where the two top logits are an exact bf16 tie in the transformers run
(gap 0.015 in a f32 reference run, which sides with the xla-rs token): the
continuations then diverge while staying equally valid.
