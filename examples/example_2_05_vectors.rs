use std::fs::File;
use std::io::Write;

use dace::*;
use ndarray_linalg::Scalar;

const ORDER: u32 = 20;

// Exercise 5.1.1: tangents and normals
fn f_da(x: &AlgebraicVector<DA>) -> DA {
    -1.0 / 3.0 * (x[0].square() + x[1].square() / 2.0) + (&x[0] / 2.0 + &x[1] / 4.0).exp()
}

fn f_f64(x: &AlgebraicVector<f64>) -> f64 {
    -1.0 / 3.0 * (x[0].square() + x[1].square() / 2.0) + (&x[0] / 2.0 + &x[1] / 4.0).exp()
}

fn ex5_1_1() {
    let mut surf = AlgebraicVector::<DA>::zeros(3);
    surf[0] = da!(1);
    surf[1] = da!(2);
    surf[2] = f_da(&surf); // trick: f() only uses the first 2 components of surf, which we already set
    let t1 = surf.deriv(1);
    let t2 = surf.deriv(2);
    let n = t1.cross(&t2).normalize(); // normalized surface normal

    println!("Exercise 5.1.1: tangents and normals\n{} {} {}", t1, t2, n);
}

// Exercise 5.1.2: (Uncontrolled) Equations of motion of the inverted pendulum
fn ode_pendulum(x: &AlgebraicVector<f64>) -> AlgebraicVector<f64> {
    // constants
    const L: f64 = 1.0; // length of pendulum (m)
    const MO: f64 = 0.1; // weight of balanced object (kg)
    const MC: f64 = 0.4; // weight of cart (kg)
    const G: f64 = 9.81; // gravity acceleration constant on earth (kg*m/s^2)

    // variables
    let sint = x[0].sin(); // sine of theta
    let cost = x[0].cos(); // cosine of theta

    // Equations of motion
    darray![
        x[1],
        ((MC + MO) * G * sint - MO * L * x[1].square() * sint * cost)
            / ((MC + MO) * L + MO * L * cost.square()),
    ]
}

fn ex5_1_2() {
    let x = darray![1.0, 0.0];
    println!("Exercise 5.1.2: Equations of Motion\n{}", ode_pendulum(&x));
}

// Exercise 5.2.1: Solar flux
fn ex5_2_1() {
    let mut surf = AlgebraicVector::<DA>::zeros(3);
    surf[0] = da!(1);
    surf[1] = da!(2);
    surf[2] = f_da(&surf); // trick: f() only uses the first 2 components of surf, which we already set
    let t1 = surf.deriv(1).normalize(); // normalizing these helps keep the coefficents small and prevents roundoff errors
    let t2 = surf.deriv(2).normalize();
    let n = t1.cross(&t2).normalize(); // normalized surface normal
    let sun = darray![0.0, 0.0, 1.0]; // sun direction
    let flux = n.dot(&sun); // solar flux on the surface

    // Output results
    println!("Exercise 5.2.1: Solar flux\n{}", flux);
    const N: i32 = 30;
    const NF: f64 = N as f64;
    let mut arg = AlgebraicVector::<f64>::zeros(2);

    let mut file = File::create("ex5_2_1.dat").unwrap();
    for i in -N..=N {
        arg[0] = (i as f64) / NF;
        for j in -N..=N {
            arg[1] = (j as f64) / NF;
            let res = surf.eval(&arg);
            writeln!(
                &mut file,
                "{}    {}    {}    {}",
                res[0],
                res[1],
                f_f64(&res),
                flux.eval(&arg)
            )
            .unwrap()
        }
        writeln!(&mut file).unwrap();
    }
}

// Exercise 5.2.2: Area
fn ex5_2_2() {
    let x = da!(1);
    let t = da!(2);
    let res = darray![
        da!(1),
        ((1.0 - x.square()) * (&t + 1.0) + (x.pow(3) - &x) * (1.0 - &t)) / 2.0,
    ];
    println!("Exercise 5.2.2: Area\n{}", res);
}

fn main() {
    DA::init(ORDER, 2); // init with maximum computation order

    ex5_1_1();
    ex5_1_2();
    ex5_2_1();
    ex5_2_2();
}
