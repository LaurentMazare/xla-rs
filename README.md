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

The [Qwen3.5-2B model](https://huggingface.co/Qwen/Qwen3.5-2B) can be used to
generate text. It is a hybrid architecture mixing gated DeltaNet linear
attention layers with full attention layers. The example loads the safetensors
weights directly and runs greedy generation with a kv-cache and DeltaNet state
carry-over.

The weights and tokenizer config are downloaded automatically from the hub
(and cached locally):

```bash
# Run the example, use --cpu to run on cpu rather than gpu.
cargo run --example qwen35 --release --features hf-hub -- \
  --prompt "What is the capital of France? Answer in one word." \
  --sample-len 30
```
