use dace::*;

fn somb(x: &AlgebraicVector<DA>) -> DA {
    let norm_x = x.vnorm();
    norm_x.sin() / norm_x
}

fn main() {
    // initialize DACE for 1st-order computations in 2 variables
    DA::init(1, 2);

    println!("Initialize x as two-dim DA vector around (2,3)\n");

    let x = darray![2.0 + da!(1), 3.0 + da!(2)];

    println!("x\n\n{x}\n");

    // Evaluate sombrero function
    let z = somb(&x);

    println!("Sombrero function\n\nz\n{z}");

    // Compute gradient of sombrero function
    let grad_z = z.gradient();

    println!("Gradient of sombrero function\n\n{grad_z}");
}
