# xla-rs
Experimentation using the xla compiler from rust

Pre-compiled binaries for the xla library can be downloaded from the
[elixir-nx/xla repo](https://github.com/elixir-nx/xla/releases/tag/v0.6.0).
These should be extracted at the root of this repository, resulting
in a `xla_extension` subdirectory being created, the currently supported version
is 0.6.0.

For a linux platform, this can be done via:
```bash
wget https://github.com/elixir-nx/xla/releases/download/v0.6.0/xla_extension-x86_64-linux-gnu-cpu.tar.gz
tar -xzvf xla_extension-x86_64-linux-gnu-cpu.tar.gz
```

The path for `xla_extension` must be specified via the `XLA_EXTENSION_DIR` environment variable.

## Generating some Text Samples with LLaMA

The [LLaMA large language model](https://github.com/facebookresearch/llama) can
be used to generate text. The model weights are only available after completing
[this form](https://forms.gle/jk851eBVbX1m5TAv5) and once downloaded can be
converted to a format this crate can use.  This requires a GPU with 16GB of
memory or 32GB of memory when running on cpu (using the -cpu flag).

```bash
# Download the tokenizer config.
wget https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json -O llama-tokenizer.json

# Extract the pre-trained weights, this requires the transformers python library to be installed.
# This creates a npz file storing all the weights.
python examples/llama/convert_checkpoint.py ..../LLaMA/7B/consolidated.00.pth

# Run the example.
cargo run --example llama --release
```

## Generating some Text Samples with GPT2

One of the featured examples is GPT2. In order to run it, one should first
download the tokenization configuration file as well as the weights before
running the example. In order to do this, run the following commands:

```bash
# Download the vocab file.
wget https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe

# Extract the pre-trained weights, this requires the transformers python library to be installed.
# This creates a npz file storing all the weights.
python examples/nanogpt/get_weights.py

# Run the example.
cargo run --example nanogpt --release
```
