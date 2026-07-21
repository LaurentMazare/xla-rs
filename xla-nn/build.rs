// The xla crate's build script links libxla_extension and embeds an rpath into
// its own targets, but `cargo:rustc-link-arg` does not propagate to dependent
// crates. Re-emit the rpath here so the binaries built in this crate (the
// examples) can find libxla_extension.so at runtime, matching the xla crate.
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=XLA_EXTENSION_DIR");
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    if os != "linux" && os != "macos" {
        return;
    }
    let xla_dir = env::var("XLA_EXTENSION_DIR")
        .map_or_else(|_| env::current_dir().unwrap().join("xla_extension"), PathBuf::from);
    let lib = xla_dir.join("lib");
    if os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    } else {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib.display());
    }
}
