use std::fs::File;
use std::io::Write;

use dace::*;
use ndarray_linalg::Scalar;

macro_rules! define_functions {
    ($t:ty) => {
        ::paste::paste! {

            // Exercise 6.1.1: The Mid-point rule integrator
            pub fn [<midpoint_ $t:lower>](
                x0: &AlgebraicVector<$t>,
                t0: f64,
                t1: f64,
                f: fn(&AlgebraicVector<$t>, f64) -> AlgebraicVector<$t>,
            ) -> AlgebraicVector<$t> {
                const HMAX: f64 = 0.005;
                let steps: f64 = ((t1 - t0) / HMAX).ceil();
                let h: f64 = (t1 - t0) / steps;
                let mut t: f64 = t0;
                let mut x: AlgebraicVector<$t> = x0.to_owned();
                let mut xmid: AlgebraicVector<$t>;
                for _i in 0..(steps as i32) {
                    xmid = &x + 0.5 * h * f(&x, t);
                    x += h * f(&xmid, t + 0.5 * h);
                    t += h;
                }
                x
            }

            // Exercise 6.1.2: Model of the (uncontrolled) pendulum
            // x = ( theta, theta_dot )
            // since the motion for x decouples we ignore it here
            fn [<pendulum_rhs_ $t:lower>](x: &AlgebraicVector<$t>, _t: f64) -> AlgebraicVector<$t> {
                // pendulum constants
                const L: f64 = 0.61; // m
                const M1: f64 = 0.21; // kg
                const M2: f64 = 0.4926; // kg
                const G: f64 = 9.81; // kg*m/s^2

                darray![
                    x[1].clone(),
                    ((M2 + M1) * G * x[0].sin() - M1 * L * x[1].square() * x[0].sin() * x[0].cos())
                        / ((M2 + M1) * L + M1 * L * x[0].cos().square()),
                ]
            }

        }
    };
}

// Create specialized functions for DA and f64 type
define_functions!(DA);
define_functions!(f64);

fn ex6_1_2() {
    let mut xda: AlgebraicVector<DA> = darray![da!(0.0), 0.2 + 0.04 * da!(1)]; // initial condition (DA)
    let mut xdb: AlgebraicVector<f64> = darray![0.0, 0.2]; // initial condition (f64)
    let mut t: f64;
    const DT: f64 = 0.05; // take a snap shot every 0.1s
    let mut file = File::create("ex6_1_2.dat").unwrap();

    t = 0.0;
    for _i in 0..100 {
        xdb = midpoint_f64(&xdb, t, t + DT, pendulum_rhs_f64); // propagate forward for dt seconds
        writeln!(&mut file, "{}   {}   {}", t, xdb[0], xdb[1]).unwrap();
        t += DT;
    }
    write!(&mut file, "\n\n").unwrap();

    t = 0.0;
    for _i in 0..100 {
        xda = midpoint_da(&xda, t, t + DT, pendulum_rhs_da); // propagate forward for dt seconds
        writeln!(
            &mut file,
            "{}   {}   {}   {}   {}",
            t,
            xda[0].eval(-1.0),
            xda[0].eval(1.0),
            xda[1].eval(-1.0),
            xda[1].eval(1.0)
        )
        .unwrap();
        t += DT;
    }
    writeln!(&mut file).unwrap();

    println!(
        "Exercise 6.1.2: Model of the (uncontrolled) pendulum\n{}",
        xda
    );
}

/// convenience routine to evaluate and plot
fn plot(x: &AlgebraicVector<DA>, t: f64, n: i32, file: &mut File) {
    let nf = n as f64;
    for i in -n..=n {
        let arg_0 = (i as f64) / nf;
        for j in -n..=n {
            let arg_1 = (j as f64) / nf;
            let res = x.eval(&darray![arg_0, arg_1]);
            writeln!(file, "{}    {}    {}", t, res[0], res[1]).unwrap();
        }
        writeln!(file).unwrap();
    }
    writeln!(file).unwrap();
}

// Exercise 6.1.3: Set propagation
// the right hand side
fn f(x: &AlgebraicVector<DA>, _t: f64) -> AlgebraicVector<DA> {
    let alpha = 0.1;
    (1.0 + alpha * x.vnorm()) * darray![-&x[1], x[0].to_owned()]
}

fn ex6_1_3() {
    let mut x = AlgebraicVector::<DA>::zeros(2);
    let mut t: f64;
    const DT: f64 = 2.0 * PI / 6.0;

    let mut file = File::create("ex6_1_3.dat").unwrap();

    // initial condition box
    x[0] = 2.0 + da!(1);
    x[1] = da!(2);
    t = 0.0;

    plot(&x, t, 7, &mut file);

    for _i in 0..6 {
        // propagate forward for dt seconds
        x = midpoint_da(&x, t, t + DT, f);
        t += DT;
        plot(&x, t, 7, &mut file);
    }

    println!("Exercise 6.1.3: Set propagation\n{}", x);
}

// Exercise 6.1.4: State Transition Matrix
fn ex6_1_4() {
    let mut x = AlgebraicVector::<DA>::zeros(2);

    // initial condition around (1,1)
    x[0] = 1.0 + da!(1);
    x[1] = 1.0 + da!(2);
    x = midpoint_da(&x, 0.0, 2.0 * PI, f);

    println!("Exercise 6.1.4: State Transition Matrix");
    println!("{}    {}", x[0].deriv(1).cons(), x[0].deriv(2).cons());
    println!("{}    {}", x[1].deriv(1).cons(), x[1].deriv(2).cons());
    println!();
}

// Exercise 6.1.5: Parameter dependence
// the right hand side (note: now it can only be evaluated with DA because alpha is a DA!)
fn f_param(x: &AlgebraicVector<DA>, _t: f64) -> AlgebraicVector<DA> {
    let alpha = 0.05 + 0.05 * da!(1); // parameter, now it's a DA
    (1.0 + alpha * x.vnorm()) * darray![-&x[1], x[0].to_owned()]
}

fn ex6_1_5() {
    let mut x = AlgebraicVector::<DA>::zeros(2);

    // initial condition (1,1)
    x[0] = 1.0.into();
    x[1] = 1.0.into();

    x = midpoint_da(&x, 0.0, 2.0 * PI, f_param);

    let mut file = File::create("ex6_1_5.dat").unwrap();
    writeln!(&mut file, "1 1").unwrap();
    for i in 0..=20 {
        let i_f = i as f64;
        writeln!(
            &mut file,
            "{}   {}",
            x[0].eval(-1.0 + i_f / 10.0),
            x[1].eval(-1.0 + i_f / 10.0)
        )
        .unwrap();
    }

    println!("Exercise 6.1.5: Parameter dependence\n{}", x);
}

// Exercise 6.2.1: 3/8 rule RK4 integrator
fn rk4(
    x0: &AlgebraicVector<DA>,
    t0: f64,
    t1: f64,
    f: fn(&AlgebraicVector<DA>, f64) -> AlgebraicVector<DA>,
) -> AlgebraicVector<DA> {
    const HMAX: f64 = 0.01;
    let steps = ((t1 - t0) / HMAX).ceil();
    let h = (t1 - t0) / steps;
    let mut t = t0;

    let mut x = x0.to_owned();
    for _i in 0..(steps as i32) {
        let k1 = f(&x, t);
        let k2 = f(&(&x + h * &k1 / 3.0), t + h / 3.0);
        let k3 = f(&(&x + h * (-&k1 / 3.0 + &k2)), t + 2.0 * h / 3.0);
        let k4 = f(&(&x + h * (&k1 - &k2 + &k3)), t + h);
        x += h * (k1 + 3.0 * k2 + 3.0 * k3 + k4) / 8.0;
        t += h;
    }

    x
}

// Exercise 6.2.2: Artsy Set Propagation
fn ex6_2_2() {
    let mut x = AlgebraicVector::<DA>::zeros(2);
    let mut t: f64;
    const DT: f64 = 2.0 * PI / 6.0;

    let mut file = File::create("ex6_2_2.dat").unwrap();

    // initial condition (c.f. example_2_05_vectors.rs)
    // (note: da1 and da2 defined just for convenience)
    let da1 = &da!(1);
    let da2 = &da!(2);
    x[0] = da1.clone();
    x[1] = ((1.0 - da1 * da1) * (da2 + 1.0) + (da1 * da1 * da1 - da1) * (1.0 - da2)) / 2.0;
    t = 0.0;
    plot(&x, t, 7, &mut file);

    for _i in 0..6 {
        x = midpoint_da(&x, t, t + DT, f); // propagate forward for dt seconds
        t += DT;
        plot(&x, t, 7, &mut file);
    }

    println!("Exercise 6.2.2: Artsy Set propagation\n{}", x);
}

// Exercise 6.2.3: CR3BP
fn cr3bp(x: &AlgebraicVector<DA>, _t: f64) -> AlgebraicVector<DA> {
    const MU: f64 = 0.30404234e-5;
    let (mut d1, mut d2): (DA, DA);

    d1 = ((&x[0] + MU).square() + x[1].square() + x[2].square()).sqrt();
    d1 = 1.0 / (&d1 * &d1 * &d1); // first distance
    d2 = ((&x[0] + MU - 1.0).square() + x[1].square() + x[2].square()).sqrt();
    d2 = 1.0 / (&d2 * &d2 * &d2); // second distance

    darray![
        x[3].clone(),
        x[4].clone(),
        x[5].clone(),
        &x[0] + 2.0 * &x[4] - &d1 * (1.0 - MU) * (&x[0] + MU) - &d2 * MU * (&x[0] + MU - 1.0),
        &x[1] - 2.0 * &x[3] - &d1 * (1.0 - MU) * &x[1] - &d2 * MU * &x[1],
        -&d1 * (1.0 - MU) * &x[2] - &d2 * MU * &x[2],
    ]
}

fn ex6_2_3() {
    const T: f64 = 3.05923;
    let x0 = darray![0.9888426847, 0.0, 0.0011210277, 0.0, 0.0090335498, 0.0];
    let mut x = x0 + AlgebraicVector::<DA>::identity(6);

    DA::push_truncation_order(1); // only first order computation needed

    x = rk4(&x, 0.0, T, cr3bp);

    println!("Exercise 6.2.3: CR3BP STM\n{}", x.cons());
    for i in 0..6 {
        for j in 1..=6 {
            print!("{}  ", x[i].deriv(j).cons());
        }
        println!();
    }
    println!();

    DA::pop_truncation_order();
}

// Exercise 6.2.4: Set propagation revisited
fn ex6_2_4() {
    let mut x = AlgebraicVector::<DA>::zeros(2);
    let mut t: f64;
    const DT: f64 = 2.0 * PI / 6.0;

    let mut file = File::create("ex6_2_4.dat").unwrap();

    // initial condition box, in polar coordinates
    x[0] = (0.3 * da!(2)).cos() * (2.0 + da!(1));
    x[1] = (0.3 * da!(2)).sin() * (2.0 + da!(1));
    t = 0.0;
    plot(&x, t, 40, &mut file);

    for _i in 0..6 {
        x = midpoint_da(&x, t, t + DT, f); // propagate forward for dt seconds
        t += DT;
        plot(&x, t, 40, &mut file);
    }

    println!("Exercise 6.2.4: Set propagation revisited\n{}", x);
}

// Exercise 6.2.5: The State Transition Matrix reloaded
fn ex6_2_5() {
    // initial condition (1, 1) plus DA identity (but in DA(2) and DA(3) as DA(1) is already used for alpha!)
    let mut x = darray![1.0 + da!(2), 1.0 + da!(3)];

    x = midpoint_da(&x, 0.0, 2.0 * PI, f_param);

    // we want to evaluate the derivatives at (alpha,0,0), so keep DA(1) and replace DA(2) and DA(3) by zero
    let mut arg = AlgebraicVector::<DA>::zeros(3);
    arg[0] = da!(1);

    println!("Exercise 6.2.5: The State Transition Matrix reloaded");
    println!("{} {}", x[0].deriv(2).eval(&arg), x[0].deriv(3).eval(&arg));
    println!("{} {}", x[1].deriv(2).eval(&arg), x[1].deriv(3).eval(&arg));
    println!();
}

fn main() {
    DA::init(15, 6); // init with maximum computation order

    ex6_1_2();
    ex6_1_3();
    ex6_1_4();
    ex6_1_5();

    ex6_2_2();
    ex6_2_3();
    ex6_2_4();
    ex6_2_5();
}
