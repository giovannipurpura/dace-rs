use dace::*;

// This requires a LAPACK binding for ndarray-linalg, see the README.
fn main() {
    DA::init(10, 1);

    let x = da!(1);

    // Compute Taylor expansion of sin(x)
    let y = x.sin();

    // Invert Taylor polynomial
    let inv_y = darray![y].invert().unwrap();

    // Compare with asin(x)
    println!("Polynomial inversion of sin(x)\n{inv_y}\n");
    println!("asin(x)\n{}", x.asin());
}
