use cmake;

fn main() {

    let dst = cmake::build("dacecore");

    println!("cargo:rerun-if-changed=dacecore");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=dacecore");
}
