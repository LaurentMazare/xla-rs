extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=xla_rs");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");

    let bindings = bindgen::Builder::default()
        .header("xla_rs/xla_rs.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("c_xla.rs"))
        .expect("Couldn't write bindings!");
}
