extern crate cbindgen;

use std::env;

fn main() {
    println!("cargo:rustc-link-arg=-rdynamic");
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let mut config: cbindgen::Config = Default::default();
    config.language = cbindgen::Language::C;
    config.include_guard = Some(String::from("__isox_host__"));
    config.parse.parse_deps = true;
    config.parse.include = Some({
        let mut includes = Vec::new();
        includes.push(String::from("isox_comms"));
        includes
    });

    cbindgen::Builder::new()
        .with_crate(crate_dir.clone())
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("c_src/isox_host_bindings.h");

    let mut config: cbindgen::Config = Default::default();
    config.language = cbindgen::Language::Cxx;
    config.include_guard = Some(String::from("__isox_host__"));
    config.parse.parse_deps = true;
    config.parse.include = Some({
        let mut includes = Vec::new();
        includes.push(String::from("isox_comms"));
        includes
    });

    cbindgen::Builder::new()
        .with_crate(crate_dir.clone())
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("c_src/isox_host_bindings.hpp");
}
