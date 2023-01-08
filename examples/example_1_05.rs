use dace::*;

fn main() {
    DA::init(20, 1);

    let x = da!(1);

    let y = x.sin();

    // compute Taylor expansion of d[sin(x)]/dx
    let dy = y.deriv(1);

    // print d[sin(x)]/dx and cos(x) to compare
    println!("d[sin(x)]/dx\n{dy}");
    println!("cos(x)\n{}", x.cos());

    // compute Taylor expansion of int[sin(x)dx]
    let int_y = y.integ(1);

    // print int[sin(x)dx] and -cos(x) to compare
    println!("int[sin(x)dx]\n{int_y}");
    println!("-cos(x)\n{}", -x.cos());
}
