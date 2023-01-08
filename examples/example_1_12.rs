use dace::*;

fn main() {
    DA::init(10, 2);

    let tol = 1.0e-12;

    let mu = 1.0;

    let mut a = da!(1.0);
    let mut e = da!(0.5);
    let t = PI / 2.0;

    let mut m = (mu / a.pow(3)).sqrt() * t; // real at this stage (i.e. constant DA)

    let mut ecc_an = m.clone(); // first guess

    let mut err = (&ecc_an - &e * ecc_an.sin() - &m).abs();

    // Newton's method for the reference solution
    while err > tol {
        ecc_an -= (&ecc_an - &e * ecc_an.sin() - &m) / (1.0 - &e * ecc_an.cos());
        err = (&ecc_an - &e * &ecc_an.sin() - &m).abs();
    }
    println!("Reference solution: E = {}\n", ecc_an.cons());

    a += da!(1);
    e += da!(2);

    m = (mu / a.pow(3)).sqrt() * t; // now M is a DA (with a non const part)

    // Newton's method for the Taylor expansion of the solution
    let mut i = 1;
    while i <= 10 {
        ecc_an -= (&ecc_an - &e * ecc_an.sin() - &m) / (1.0 - &e * ecc_an.cos());
        i *= 2;
    }

    println!("Taylor expansion of E\n{}\n", ecc_an);

    println!("Let's verify it is the Taylor expansion of the solution:");
    println!("Evaluate (E - e*sin(E) - M) in DA");

    println!("{}", &ecc_an - &e * ecc_an.sin() - &m);
}
