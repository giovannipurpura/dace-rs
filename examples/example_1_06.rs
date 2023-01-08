use dace::*;

fn err_func(x: &DA) -> DA {
    // Note: this is not the classical erf
    1.0 / (2.0 * PI).sqrt() * (-x.sqr() / 2.0).exp()
}

fn main() {
    // initialize DACE for 24th-order computations in 1 variable
    DA::init(24, 1);

    let x = da!(1);

    // compute Taylor expansion of 1/sqrt(2 * pi) * exp(-x^2/2)
    let y = err_func(&x);

    // compute the Taylor expansion of the indefinite integral of
    // 1/sqrt(2 * pi) * exp(-x^2/2)
    let int_y = y.integ(1);

    // compute int_{-1}^{+1} 1/sqrt(2 * pi) * exp(-x^2/2) dx
    let value = int_y.eval(1.0) - int_y.eval(-1.0);

    println!("int_{{-1}}^{{+1}} 1/sqrt(2 * pi) * exp(-x^2/2) dx");
    println!("Exact result: 0.682689492137");
    println!("Approx. result: {value:.12}");
    println!(
        "Equivalent using DACE: {:.12}",
        da!(0.5).sqrt().erf().cons()
    );
}
