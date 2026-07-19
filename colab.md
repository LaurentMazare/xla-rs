# Running the Qwen3.5 example on a Colab TPU

These steps build `xla-rs` and run the `qwen35` example on a TPU-enabled Google
Colab notebook. The TPU PjRt client and the device auto-detection are already on
`main`, so a plain clone is enough — no source edits required.

The one thing that matters beyond the build: the **`libtpu.so` PJRT version must
be at least as new as the extension's framework version** (see step 6). Colab's
stock libtpu is often too old; upgrading it to the latest is what makes this
work.

Run each block in a terminal, or in notebook cells (prefix with `!` / use a
`%%bash` cell).

## 0. Select the TPU runtime

Runtime → Change runtime type → **TPU**.

## 1. Confirm the TPU is live

```bash
python3 -c "import jax; print(jax.devices())"
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
```

## 6. Get a new-enough libtpu (the key step)

The extension is compiled against a specific PJRT C-API version — its
"framework" version (`0.90` for xla_extension 0.10.0). The `libtpu.so` you run
against must report a PJRT API version **>= that framework version**, otherwise
client init or HLO compilation fails. Two traps:

- The `libtpu.so` **bundled** inside the extension (`xla_extension/lib/`) is
  stale build scaffolding (PJRT `0.38`). Do **not** use it — it's too old for
  both the extension and Colab's TPU driver (`Couldn't allocate MSIX interrupts`).
- Colab's **stock** pip libtpu can also lag the framework (e.g. PJRT `0.75` vs
  `0.90`), which surfaces as `reshape.NNN instruction contains invalid operand id(s)`.

So force-install the **latest** pip libtpu, which is newer than the framework
(`0.0.44` reports PJRT `0.113`):

```bash
pip install -q --force-reinstall libtpu==0.0.44
```

Optionally verify it (reads the PJRT version straight out of the `.so`):

```bash
python3 - <<'PY'
import ctypes
so = "/usr/local/lib/python3.12/dist-packages/libtpu/libtpu.so"
lib = ctypes.CDLL(so); lib.GetPjrtApi.restype = ctypes.c_void_p
p = lib.GetPjrtApi(); maj, mino = (ctypes.c_int * 2).from_address(p + 32)
print(f"{so}: PJRT API {maj}.{mino}")   # want >= framework (0.90 for xla 0.10.0)
PY
```

## 7. Build and run

```bash
export XLA_EXTENSION_DIR=/content/xla_extension
export TPU_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/libtpu/libtpu.so
cd /content/xla-rs
cargo run --example qwen35 --release --features hf-hub -- \
  --which 2b --prompt "What is the capital of France?" --sample-len 200
```

No `--tpu` flag is needed: the example's `make_client` tries the TPU client
first and falls back to GPU/CPU only if it is unavailable. On success it prints
`platform: tpu ...`. Use `--which 2b` for a quick first test (single weight
shard); weights download from the Hugging Face hub on first run and are cached.

## Troubleshooting

- **`Couldn't allocate MSIX interrupts` / `IRQ ... NORESIZE`** — the libtpu is
  too old (or too new) for Colab's TPU driver. The latest pip libtpu (step 6)
  matches current Colab.
- **`reshape.NNN instruction contains invalid operand id(s)`** — the libtpu's
  PJRT/HLO is older than the extension's; upgrade libtpu (step 6).
- **`InitGoogle() has not finished yet` abort** — an old/mismatched libtpu; use
  the latest pip one.
- **TPU already in use** — a crashed run can hold the chip. Free it with
  `sudo fuser -k /dev/vfio/* /dev/accel* 2>/dev/null; sudo rm -f /tmp/libtpu_lockfile`,
  or Runtime → Restart session (keeps everything under `/content`). Run the Rust
  binary *before* any `import jax` cell, since that also grabs the TPU.
- The device string passed to `GetCApiClient` is upper-case `"TPU"` (matching
  EXLA); the plugin is loaded/initialized as lower-case `"tpu"`.
```
