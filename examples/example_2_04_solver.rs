use std::fs::File;
use std::io::Write;

use dace::*;

const ORDER: u32 = 10;

fn nf_da(x: &DA, p: &DA) -> DA {
    x - (x * x - p) / (2.0 * x)
}

fn nf_f64(x: f64, p: f64) -> f64 {
    x - (x * x - p) / (2.0 * x)
}

// Exercise 4.1.3: Naive Newton
fn ex4_1_3() {
    const TOL: f64 = 1e-14; // tolerance
    let p0 = 4.0;
    let x0 = 1.0; // expansion point and initial guess
    let p = p0 + da!(1); // DA parameter
    let mut x = da!(x0); // DA initial guess
    let mut xp: DA;
    let mut i = 0;

    loop {
        xp = x;
        x = nf_da(&xp, &p);
        i += 1;
        if (&xp - &x).abs() <= TOL || i >= 1000 {
            break;
        }
    }

    println!("Exercise 4.1.3: Naive Newton\n{}\n{}", x, p.sqrt() - &x);
    println!("Number of iterations: {i}\n");
}

// Exercise 4.1.4: complicated parameters
fn ex4_1_4() {
    let p0 = 0.0;
    let x0 = 1.0; // x0 must now satisfy f(x0,cos(p0))=0
    let p = (p0 + da!(1)).cos();
    let mut x = da!(x0);

    let mut i = 1;
    while i <= ORDER {
        x = nf_da(&x, &p);
        i *= 2;
    }
    println!(
        "Exercise 4.1.4: Complicated parameters\n{}\n{}",
        x,
        p.sqrt() - &x
    );
}

// Exercise 4.2.1: Full DA Newton solver
fn ex4_2_1(p0: f64) {
    const TOL: f64 = 1e-14;
    let mut x0 = p0 / 2.0;
    let mut xp: f64; // x0 is just some initial guess
    let mut i = 0;

    // double precision computation => fast
    loop {
        xp = x0;
        x0 = nf_f64(xp, p0);
        i += 1;
        if (xp - x0).abs() <= TOL || i >= 1000 {
            break;
        }
    }

    // DA computation => slow
    let p = p0 + da!(1);
    let mut x = da!(x0);
    let mut i = 1;
    while i <= ORDER {
        x = nf_da(&x, &p);
        i *= 2;
    }

    println!("Exercise 4.2.1: Full DA Newton\n{}\n{}", x, p.sqrt() - &x);
}

// Exercise 4.2.2 & 4.2.3: Kepler's equation solver
fn nkep_da(e: &DA, m: &DA, ecc: &DA) -> DA {
    e - (e - ecc * e.sin() - m) / (1.0 - ecc * e.cos())
}

fn nkep_f64(e: f64, m: f64, ecc: f64) -> f64 {
    e - (e - ecc * e.sin() - m) / (1.0 - ecc * e.cos())
}

// double precision Kepler solver
fn kepler(m: f64, ecc: f64) -> f64 {
    const TOL: f64 = 1e-14;
    let mut e0 = m;
    let mut ep: f64;
    let mut i = 0;

    loop {
        ep = e0;
        e0 = nkep_f64(ep, m, ecc);
        i += 1;
        if (ep - e0).abs() <= TOL || i >= 1000 {
            break;
        }
    }

    e0
}

fn ex4_2_2(m0: f64, ecc0: f64) {
    let m = m0 + da!(1);
    let mut e = da!(kepler(m0, ecc0)); // reference solution
    let ecc = da!(ecc0); // keep eccentricity constant (4.2.2)
    // let ecc = ecc0 + 0.1 * da!(2); // also expand w.r.t. eccentricity (4.2.3)

    let mut i = 1;
    while i <= ORDER {
        e = nkep_da(&e, &m, &ecc);
        i *= 2;
    }

    println!("Exercise 4.2.2: Expansion of the Anomaly");
    println!("Resulting expansion:\n{}", e);
    println!("Residual error:\n{}", (&e - ecc * e.sin() - &m));

    // sample the resulting polynomial over M0+-3 rad
    let mut file = File::create("ex4_2_2.dat").unwrap();
    for i in -300..300 {
        let i_100 = (i as f64) / 100.0;
        writeln!(
            &mut file,
            "{}    {}    {}",
            m0 + i_100,
            e.eval(i_100),
            kepler(m0 + i_100, ecc0)
        )
        .unwrap();
    }
    // gnuplot command: plot 'ex4_2_2.dat' u ($1*180/pi):($2*180/pi) w l t 'DA', 'ex4_2_2.dat' u ($1*180/pi):($3*180/pi) w l t 'pointwise'
}

fn main() {
    DA::init(ORDER, 2); // init with maximum computation order

    ex4_1_3();
    ex4_1_4();
    ex4_2_1(9.0);
    ex4_2_2(0.0, 0.5);
}
