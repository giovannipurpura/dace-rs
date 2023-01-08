use dace::*;
use ndarray::{concatenate, s, Axis};
use std::time::Instant;

const MU: f64 = 398600.0; // km^3/s^2

fn tbp(x: &AlgebraicVector<DA>, _t: f64) -> AlgebraicVector<DA> {
    let pos = x.slice(s![0..3]);
    let vel = x.slice(s![3..6]);

    let r = AlgebraicVector::from(pos).vnorm();
    let acc = -&pos * MU / r.pow(3);

    concatenate![Axis(0), vel, acc].into()
}

fn euler(x: &AlgebraicVector<DA>, t0: f64, t1: f64) -> AlgebraicVector<DA> {
    let hmax = 0.1;
    let steps = ((t1 - t0) / hmax).ceil();
    let h = (t1 - t0) / steps;
    let mut t = t0;

    let mut x = x.to_owned();
    for _ in 0..steps as u32 {
        x += h * tbp(&x, t);
        t += h;
    }
    x
}

fn main() {
    DA::init(3, 6);

    // Set initial conditions
    let ecc = 0.5;

    let mut x0 = AlgebraicVector::identity(6);
    x0[0] += 6678.0; // 300 km altitude
    x0[4] += (MU / 6678.0 * (1.0 + ecc)).sqrt();

    // integrate for half the orbital period
    let a = 6678.0 / (1.0 - ecc);

    let start_insant = Instant::now();
    let xf = euler(&x0, 0.0, PI * (a.powi(3) / MU).sqrt());

    println!("Initial conditions:\n{}\n", x0);
    println!("Final conditions:\n{}\n", xf);
    println!("Initial conditions (cons. part):\n{}\n", x0.cons());
    println!("Final conditions: (cons. part)\n{}\n", xf.cons());

    // Evaluate for a displaced initial condition
    let delta_x0: AlgebraicVector<f64> = darray![1.0, -1.0, 0.0, 0.0, 0.0, 0.0]; // km

    let x0_cons = x0.cons();

    println!("Displaced Initial condition:\n{}\n", x0_cons + &delta_x0);
    println!("Displaced Final condition:\n{}\n", xf.eval(&delta_x0));

    println!(
        "Info: time required for integration = {:?} s",
        start_insant.elapsed()
    );
}
