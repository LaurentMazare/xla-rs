# Running the Qwen3.5 example on a Colab TPU

These steps build `xla-rs` and run the `qwen35` example on a TPU-enabled Google
Colab notebook. Everything needed (the TPU PjRt client and the device
auto-detection) is already on `main`, so a plain clone is enough — no source
edits required.

Run each block in a notebook cell (prefix shell commands with `!`, or use a
`%%bash` cell).

## 0. Select the TPU runtime

Runtime → Change runtime type → **TPU**.

## 1. Confirm the TPU is live and locate `libtpu.so`

JAX is preinstalled on the TPU runtime and pulls in `libtpu`.

```bash
python3 -c "import jax; print(jax.devices())"
find / -name 'libtpu.so' 2>/dev/null | head        # note this path for step 6
```

## 2. Install the Rust toolchain

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

## 3. Install the build dependency for bindgen

```bash
apt-get update && apt-get install -y clang libclang-dev pkg-config
```

## 4. Download the XLA TPU extension (v0.10.0, x86_64 host)

```bash
cd /content
wget https://github.com/elixir-nx/xla/releases/download/v0.10.0/xla_extension-0.10.0-x86_64-linux-gnu-tpu.tar.gz
tar -xzf xla_extension-0.10.0-x86_64-linux-gnu-tpu.tar.gz     # -> /content/xla_extension
```

## 5. Clone the repository

```bash
cd /content
git clone https://github.com/LaurentMazare/xla-rs.git
cd xla-rs
```

## 6. Build and run

```bash
export XLA_EXTENSION_DIR=/content/xla_extension
export TPU_LIBRARY_PATH=$(find / -name 'libtpu.so' 2>/dev/null | head -1)

cargo run --example qwen35 --release --features hf-hub -- \
  --which 2b --prompt "What is the capital of France?" --sample-len 200
```

No `--tpu` flag is needed: the example's `make_client` tries the TPU client
first and only falls back to GPU/CPU if the TPU runtime is unavailable. On
success it prints `platform: tpu ...`. Use `--which 2b` for a quick first test
(single weight shard); the weights download from the Hugging Face hub on first
run and are cached afterwards.

## Caveats

- **Runtime-untested.** The TPU shim (`pjrt::LoadPjrtPlugin` +
  `xla::GetCApiClient("tpu")` in `xla_rs/xla_rs.cc`) compiles and links, but has
  not been exercised on real TPU hardware. `GetCApiClient("tpu")` may need
  further tweaks (e.g. `InitializePjrtPlugin`, create-options).
- **`libtpu` version compatibility** is the main risk: the `libtpu` bundled by
  Colab's JAX must be PJRT C-API-compatible with the elixir-nx XLA 0.10.0 build.
  If client creation errors, pin a matching version (`pip install libtpu==<ver>`)
  and re-point `TPU_LIBRARY_PATH`.
- If a shared library fails to load at runtime, add its directory to
  `LD_LIBRARY_PATH` (the CUDA build needed the same for NCCL/nvshmem).
