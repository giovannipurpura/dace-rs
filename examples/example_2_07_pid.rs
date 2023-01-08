use dace::*;
use ndarray_linalg::Scalar;
use std::fs::File;
use std::io::Write;

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

            // Exercise 7.1.1: Model of the controlled pendulum
            // x = ( theta, theta_dot, u )
            fn [<pendulum_rhs_ $t:lower>](x: &AlgebraicVector<$t>, _t: f64) -> AlgebraicVector<$t> {
                // pendulum constants
                const L: f64 = 0.61; // m
                const M1: f64 = 0.21; // kg
                const M2: f64 = 0.4926; // kg
                const G: f64 = 9.81; // kg*m/s^2

                let mut res = AlgebraicVector::<$t>::zeros(3);
                let (sint, cost) = (&x[0].sin(), &x[0].cos());

                res[0] = x[1].clone();
                res[1] = (&x[2] + (M1 + M2) * G * sint - M1 * L* x[1].square() * sint * cost)
                    / ((M1 + M2) * L + M1 * L * cost.square());
                // u (res[2]) is assumed constant unless changed externally by the controller

                res
            }

            // Exercise 6.2.1: 3/8 rule RK4 integrator
            fn [<rk4_ $t:lower>](
                x0: &AlgebraicVector<$t>,
                t0: f64,
                t1: f64,
                f: fn(&AlgebraicVector<$t>, f64) -> AlgebraicVector<$t>,
            ) -> AlgebraicVector<$t> {
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
        }
    };
}

// Create specialized functions for DA and f64 type
define_functions!(DA);
define_functions!(f64);

// Exercise 7.1.2: Tuning the PID simulator (double)
fn ex7_1_2() {
    // PID parameters
    const KP: f64 = 8.0;
    const TI: f64 = 3.0;
    const TD: f64 = 0.3;
    const SETPT: f64 = 0.0; // set point
    const DT: f64 = 0.05; // controller time step (50 ms)
    // const UMAX: f64 = 10.0; // maximum control (Exercise 7.2.1)

    let mut file = File::create("ex7_1_2.dat").unwrap();

    // PID controller variables
    let mut int_err = 0.0;
    let mut last_err = 0.0;

    // Initial condition (u(0)=x[2]=0)
    let mut t = 0.0;
    let mut x = darray![0.1, 0.0, 0.0];

    // propagate the model for 100 sec
    while t < 100.0 && x[0].abs() < 1.5 {
        // compute the PID control at this controller time step
        let err = SETPT - x[0];
        let derr = (err - last_err) / DT;
        int_err += last_err * DT;
        last_err = err;
        x[2] = KP * (err + TD * derr + int_err / TI);
        // prevent control saturation (Exercise 7.2.1)
        // x[2] = ((x[2] / 2.0 / UMAX).tanh() * UMAX * 2.0)
        //     .min(UMAX)
        //     .max(-UMAX);

        // output and propagate one time step
        writeln!(&mut file, "{}   {}   {}", t, x[0], x[2]).unwrap();
        x = rk4_f64(&x, t, t + DT, pendulum_rhs_f64);
        t += DT;
    }

    println!("Final angle theta: {}", x);
    if x[0].abs() > 1.5 {
        println!("WHOOPSY: Fell over after {} seconds.", t);
    }
}

// Exercise 7.1.3: PID simulator (DA)
fn ex7_1_3() {
    // PID parameters
    const KP: f64 = 8.0;
    const TI: f64 = 3.0;
    const TD: f64 = 0.3;
    const SETPT: f64 = 0.0; // set point
    const DT: f64 = 0.05; // controller time step (50 ms)
    // const UMAX: f64 = 10.0; // maximum control (Exercise 7.2.1)

    let mut file = File::create("ex7_1_3.dat").unwrap();

    // PID controller variables
    let mut int_err = DA::new();
    let mut last_err = DA::new();
    let mut t = 0.0;

    // Initial condition
    let mut x = AlgebraicVector::<DA>::zeros(3);
    x[0] = 0.1 + 0.1 * da!(1);

    // propagate the model state for 100 sec
    while t < 40.0 && x[0].cons().abs() < 1.5 {
        // compute the PID control
        let err = SETPT - &x[0];
        let derr = (&err - &last_err) / DT;
        int_err += &last_err * DT;
        last_err = err.clone();
        x[2] = KP * (&err + TD * &derr + &int_err / TI);
        // prevent control saturation (Exercise 7.2.1)
        // x[2] = (&x[2] / UMAX).tanh() * UMAX;

        // output and propagate one time step (Exercise 7.1.4)
        let bx = x[0].bound();
        let bu = x[2].bound();
        writeln!(
            &mut file,
            "{}   {}   {}   {}   {}   {}   {}   {}   {}   {}   {}",
            t,
            x[0].cons(),
            bx.m_ub,
            bx.m_lb,
            x[0].eval(-1.0),
            x[0].eval(1.0),
            x[2].cons(),
            bu.m_ub,
            bu.m_lb,
            x[2].eval(-1.0),
            x[2].eval(1.0)
        )
        .unwrap();
        x = rk4_da(&x, t, t + DT, pendulum_rhs_da);
        t += DT;
    }

    println!("Final angle theta:{}", x);
    if x[0].cons().abs() > 1.5 {
        println!("WHOOPSY: Fell over after {} seconds.", t);
    }
}

// Exercise 7.1.5: Bounding
fn ex7_1_5() {
    let x = da!(1);
    let y = da!(2);
    let func = (&x / 2.0).sin() / (2.0 + (&y / 2.0 + &x * &x).cos());

    let (mut a, b, mut c): (Interval, Interval, Interval);

    // bound by rasterizing
    let mut arg = AlgebraicVector::<f64>::zeros(2);
    a = Interval {
        m_lb: 9999999.0,
        m_ub: -9999999.0,
    };
    c = a;
    for i in -10..10 {
        arg[0] = (i as f64) / 10.0;
        for j in -10..10 {
            arg[1] = (j as f64) / 10.0;
            // polynomial expansion
            let mut r = func.eval(&arg);
            a.m_lb = a.m_lb.min(r);
            a.m_ub = a.m_ub.max(r);
            // actual function
            r = (&arg[0] / 2.0).sin() / (2.0 + (&arg[1] / 2.0 + &arg[0] * &arg[0]).cos());
            c.m_lb = c.m_lb.min(r);
            c.m_ub = c.m_ub.max(r);
        }
    }

    // DA bounding
    b = func.bound();

    println!("func:\n{}", func);
    println!("Bounds:");

    println!("DA bound:       [{}, {}]", b.m_lb, b.m_ub);
    println!("DA raster:      [{}, {}]", a.m_lb, a.m_ub);
    println!("double raster:  [{}, {}]", c.m_lb, c.m_ub);
}

// Exercise 7.2.2: PID simulator with uncertain mass (DA)

// Model of controlled pendulum with uncertain mass
// x = ( theta, theta_dot, u )
fn pendulum_rhs_mass(x: &AlgebraicVector<DA>, _t: f64) -> AlgebraicVector<DA> {
    // pendulum constants
    const L: f64 = 0.61; // m
    let m = 0.21 * (1.0 + 0.1 * da!(1)); // kg
    const M: f64 = 0.4926; // kg
    const G: f64 = 9.81; // kg*m/s^2

    let mut res = AlgebraicVector::<DA>::zeros(3);
    let (sint, cost) = (x[0].sin(), x[0].cos());

    res[0] = x[1].clone();
    res[1] = (&x[2] + (M + &m) * G * &sint - &m * L * x[1].sqr() * &sint * &cost)
        / ((M + &m) * L + &m * L * cost.sqr());
    // res[2] = 0.0; // u is assumed constant unless changed externally by the controller

    res
}

fn ex7_2_2() {
    // PID parameters
    const KP: f64 = 8.0;
    const TI: f64 = 3.0;
    const TD: f64 = 0.3;
    const SETPT: f64 = 0.0; // set point
    const DT: f64 = 0.05; // controller time step (50 ms)
    // const UMAX: f64 = 10.0; // maximum control (Exercise 7.2.1)

    let mut file = File::create("ex7_2_2.dat").unwrap();

    // PID controller variables
    let mut int_err = DA::new();
    let mut last_err = DA::new();
    let mut t = 0.0;

    // Initial condition
    let mut x: AlgebraicVector<DA> = darray![0.1, 0.0, 0.0].into();

    // propagate the model state for 100 sec
    while t < 40.0 && x[0].cons().abs() < 1.5 {
        // compute the PID control
        let err = SETPT - &x[0];
        let derr = (&err - &last_err) / DT;
        int_err += &last_err * DT;
        last_err = err.clone();
        x[2] = KP * (&err + TD * &derr + &int_err / TI);
        // prevent control saturation (Exercise 7.2.1)
        // x[2] = (&x[2] / UMAX).tanh() * UMAX;

        // output and propagate one time step (Exercise 7.1.4)
        let bx = x[0].bound();
        let bu = x[2].bound();
        writeln!(
            &mut file,
            "{}   {}   {}   {}   {}   {}   {}   {}   {}   {}   {}",
            t,
            x[0].cons(),
            bx.m_ub,
            bx.m_lb,
            x[0].eval(-1.0),
            x[0].eval(1.0),
            x[2].cons(),
            bu.m_ub,
            bu.m_lb,
            x[2].eval(-1.0),
            x[2].eval(1.0)
        )
        .unwrap();
        x = rk4_da(&x, t, t + DT, pendulum_rhs_mass);
        t += DT;
    }

    println!("Final angle theta:{}", x);
    if x[0].cons().abs() > 1.5 {
        println!("WHOOPSY: Fell over after {} seconds.", t);
    }
}

fn main() {
    DA::init(10, 2);

    ex7_1_2();
    ex7_1_3();
    ex7_1_5();

    ex7_2_2();
}
