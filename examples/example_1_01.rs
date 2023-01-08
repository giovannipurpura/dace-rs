use dace::*;

fn main() {
    // initialize DACE for 20th-order computations in 1 variable
    DA::init(20, 2);

    // initialize x as DA
    let x = da!(1);

    // compute y = sin(x)
    let y = x.sin();

    // print x and y to screen
    println!("x\n{x}");
    println!("y = sin(x)\n{}", y);
}
