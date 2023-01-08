use dace::*;

fn somb(x: &AlgebraicVector<DA>) -> DA {
    let norm_x = x.vnorm();
    norm_x.sin() / norm_x
}

fn main() {
    // initialize DACE for 10th-order computations in 2 variables
    DA::init(10, 2);

    println!("Initialize x as two-dim DA vector around (2, 3)\n");

    let x = darray![2.0 + da!(1), 3.0 + da!(2)];

    println!("x\n{x}\n");

    // Evaluate sombrero function
    let z = somb(&x);

    println!("Sombrero function\n\nz\n{z}");
}
