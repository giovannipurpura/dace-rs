use dace::*;
use ndarray::{concatenate, s, Axis};

const MU: f64 = 398600.0; // km^3/s^2

fn tbp(x: &AlgebraicVector<f64>, _t: f64) -> AlgebraicVector<f64> {
    let pos = x.slice(s![0..3]);
    let vel = x.slice(s![3..6]);

    let r = AlgebraicVector::from(pos).vnorm();
    let acc = -&pos * MU / r.powi(3);

    concatenate![Axis(0), vel, acc].into()
}

fn euler(x: &AlgebraicVector<f64>, t0: f64, t1: f64) -> AlgebraicVector<f64> {
    const HMAX: f64 = 0.1;
    let steps = ((t1 - t0) / HMAX).ceil();
    let h = (t1 - t0) / steps;
    let mut t = t0;

    let mut x = x.to_owned();
    for _ in 0..steps as i64 {
        x += h * tbp(&x, t);
        t += h;
    }
    x
}

fn main() {
    // Set initial conditions
    let ecc = 0.5;

    let mut x0 = AlgebraicVector::<f64>::zeros(6);
    x0[0] += 6678.0; // 300 km altitude
    x0[4] += (MU / 6678.0 * (1.0 + ecc)).sqrt();

    // integrate for half the orbital period
    let a = 6678.0 / (1.0 - ecc);
    let xf = euler(&x0, 0.0, PI * (a.powi(3) / MU).sqrt());

    println!("Initial conditions:\n{}\n", x0);
    println!("Final conditions:\n{}\n", xf);
}
