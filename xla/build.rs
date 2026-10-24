extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Eq, PartialEq)]
enum OS {
    Linux,
    #[allow(clippy::enum_variant_names)]
    MacOS,
    Windows,
}

impl OS {
    fn get() -> Self {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" => Self::Linux,
            "macos" => Self::MacOS,
            "windows" => Self::Windows,
            os => panic!("Unsupported system {os}"),
        }
    }
}

fn make_shared_lib<P: AsRef<Path>>(os: OS, xla_dir: P, out_dir: &Path) {
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    let mut build = cc::Build::new();
    if let Some(overlay) = patched_headers(xla_dir.as_ref(), out_dir) {
        build.include(overlay);
    }
    match os {
        OS::Linux | OS::MacOS => {
            build
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .flag("-std=c++17")
                .flag("-Wno-deprecated-declarations")
                .flag("-DLLVM_ON_UNIX=1")
                .flag("-DLLVM_VERSION_STRING=")
                // The pre-built xla extension library is compiled in release
                // mode. Some headers, e.g. tsl/concurrency/async_value.h,
                // declare additional struct members in debug builds so NDEBUG
                // has to be set to avoid ABI mismatches (which result in
                // segfaults when running computations).
                .flag("-DNDEBUG")
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
        OS::Windows => {
            build
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .flag("/DNDEBUG")
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
    };
}

/// The pre-built headers are compiled with clang upstream and some of them do
/// not go through gcc. Rather than requiring clang, the offending headers are
/// copied with a minimal fix into an include directory that is searched before
/// the xla_extension one. Returns that directory when a header was patched.
///
/// xla_extension 0.11.0, `xla/tsl/concurrency/future.h`: the const& overload
/// of `Future<T>::Flatten` is declared with `!is_move_only` in its enable_if
/// but defined with `!Future<T>::is_move_only`; gcc does not consider the two
/// expressions equivalent and rejects the definition ("no declaration
/// matches"). Both are spelled as the underlying trait, which is what
/// `Future<T>::is_move_only` evaluates to.
fn patched_headers(xla_dir: &Path, out_dir: &Path) -> Option<PathBuf> {
    let rel = Path::new("xla/tsl/concurrency/future.h");
    let src = xla_dir.join("include").join(rel);
    println!("cargo:rerun-if-changed={}", src.display());
    let content = std::fs::read_to_string(&src).ok()?;
    let decl = "std::enable_if_t<internal::IsFuture<U>::value &&\n                                             !is_move_only>* = nullptr>";
    let def = "std::enable_if_t<internal::IsFuture<U>::value &&\n                                       !Future<T>::is_move_only>*>";
    if !content.contains(decl) || !content.contains(def) {
        return None;
    }
    let trait_ = "internal::IsMoveOnly<T>::value";
    let patched = content
        .replace(decl, &decl.replace("!is_move_only", &format!("!{trait_}")))
        .replace(def, &def.replace("!Future<T>::is_move_only", &format!("!{trait_}")));
    let overlay = out_dir.join("include");
    let dst = overlay.join(rel);
    std::fs::create_dir_all(dst.parent().unwrap()).ok()?;
    std::fs::write(&dst, patched).ok()?;
    Some(overlay)
}

fn env_var_rerun(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok()
}

fn main() {
    let os = OS::get();
    let xla_dir = env_var_rerun("XLA_EXTENSION_DIR")
        .map_or_else(|| env::current_dir().unwrap().join("xla_extension"), PathBuf::from);

    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    let bindings = bindgen::Builder::default()
        .header("xla_rs/xla_rs.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("c_xla.rs")).expect("Couldn't write bindings!");

    // Exit early on docs.rs as the C++ library would not be available.
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }
    make_shared_lib(os, &xla_dir, &out_path);
    // The --copy-dt-needed-entries -lstdc++ are helpful to get around some
    // "DSO missing from command line" error
    // undefined reference to symbol '_ZStlsIcSt11char_traitsIcESaIcEERSt13basic_ostreamIT_T0_ES7_RKNSt7__cxx1112basic_stringIS4_S5_T1_EE@@GLIBCXX_3.4.21'
    if os == OS::Linux {
        // println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
        println!("cargo:rustc-link-arg=-Wl,-lstdc++");
    }
    println!("cargo:rustc-link-search=native={}", xla_dir.join("lib").display());
    println!("cargo:rustc-link-lib=static=xla_rs");
    if os == OS::MacOS {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", xla_dir.join("lib").display());
    } else {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", xla_dir.join("lib").display());
    }
    println!("cargo:rustc-link-lib=xla_extension");
}
