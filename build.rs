extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};

fn make_shared_lib<P: AsRef<Path>>(xla_dir: P) {
    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    match os.as_str() {
        "linux" | "macos" => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .flag("-std=c++14")
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
        "windows" => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
        _ => panic!("Unsupported OS"),
    };
}

fn main() {
    let xla_dir = env::current_dir().unwrap().join("xla_extension");
    make_shared_lib(&xla_dir);

    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    let bindings = bindgen::Builder::default()
        .header("xla_rs/xla_rs.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("c_xla.rs")).expect("Couldn't write bindings!");

    // The --copy-dt-needed-entries -lstdc++ are helpful to get around some
    // "DSO missing from command line" error
    // undefined reference to symbol '_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@@GLIBCXX_3.4.21'
    println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
    println!("cargo:rustc-link-arg=-Wl,-lstdc++");
    println!("cargo:rustc-link-search=native={}", xla_dir.join("lib").display());
    println!("cargo:rustc-link-lib=static=xla_rs");
    println!("cargo:rustc-link-arg=-Wl,-rpath={}", xla_dir.join("lib").display());
    println!("cargo:rustc-link-lib=xla_extension");
}
