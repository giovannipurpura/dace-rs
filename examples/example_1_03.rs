use dace::*;

fn main() {
    // Initialize DACE for 1st-order computations in 1 variable
    DA::init(1, 1);

    // Initialize x as DA around 3
    let x = 3.0 + da!(1);

    println!("x\n{x}");

    // Evaluate f(x) = 1/(x+1/x)
    let f = 1.0 / (&x + 1.0 / &x);

    println!("f(x) = 1/(x+1/x)\n{f}");
}
