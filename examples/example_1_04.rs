use dace::*;

fn main() {
    DA::init(20, 1);

    let x = da!(1);

    // Compute [cos(x)-1]
    let y = x.cos() - 1.0;

    println!("[cos(x)-1]\n{y}");

    // Compute [cos(x)-1]^11
    let z = y.pow(11);

    println!("[cos(x)-1]^11\n{z}");
}
