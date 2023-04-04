# xla-rs
Experimentation using the xla compiler from rust

Pre-compiled binaries for the xla library can be downloaded from the
[elixir-nx/xla repo](https://github.com/elixir-nx/xla/releases/tag/v0.4.4).
These should be extracted at the root of this repository, resulting
in a `xla_extension` subdirectory being created, the currently supported version
is 0.4.4.

For a linux platform, this can be done via:
```bash
wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-x86_64-linux-gnu-cpu.tar.gz
tar -xzvf xla_extension-x86_64-linux-gnu-cpu.tar.gz
```
