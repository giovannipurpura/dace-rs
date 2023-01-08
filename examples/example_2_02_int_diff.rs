use dace::*;

// Exercise 2.1.1: derivatives
fn somb_da(x: &DA, y: &DA) -> DA {
    let r = (x * x + y * y).sqrt();
    r.sin() / r
}

fn somb_f(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    r.sin() / r
}

fn ex2_1_1() {
    let x0 = 2.0;
    let y0 = 3.0;
    let x = da!(1);
    let y = da!(2);
    let func = somb_da(&(x0 + x), &(y0 + y)); // expand sombrero function around (x0,y0)

    // compute the derivative using DA
    let dadx = func.deriv(1).cons();
    let dady = func.deriv(2).cons();
    let dadxx = func.deriv(1).deriv(1).cons();
    let dadxy = func.deriv(1).deriv(2).cons();
    let dadyy = func.deriv(2).deriv(2).cons();
    let dadxxx = func.deriv(1).deriv(1).deriv(1).cons();

    // compute the derivatives using divided differences
    let h = 1e-3;
    let dx = (somb_f(x0 + h, y0) - somb_f(x0 - h, y0)) / (2.0 * h);
    let dy = (somb_f(x0, y0 + h) - somb_f(x0, y0 - h)) / (2.0 * h);
    let dxx = (somb_f(x0 + 2.0 * h, y0) - 2.0 * somb_f(x0, y0) + somb_f(x0 - 2.0 * h, y0))
        / (4.0 * h * h);
    let dxy = (somb_f(x0 + h, y0 + h) - somb_f(x0 - h, y0 + h) - somb_f(x0 + h, y0 - h)
        + somb_f(x0 - h, y0 - h))
        / (4.0 * h * h);
    let dyy = (somb_f(x0, y0 + 2.0 * h) - 2.0 * somb_f(x0, y0) + somb_f(x0, y0 - 2.0 * h))
        / (4.0 * h * h);
    let dxxx = (somb_f(x0 + 3.0 * h, y0) - 3.0 * somb_f(x0 + h, y0) + 3.0 * somb_f(x0 - h, y0)
        - somb_f(x0 - 3.0 * h, y0))
        / (8.0 * h * h * h);

    println!("Exercise 2.1.1: Numerical derivatives\n");
    println!("d/dx:    {:.15e}", (dadx - dx).abs());
    println!("d/dy:    {:.15e}", (dady - dy).abs());
    println!("d/dxx:   {:.15e}", (dadxx - dxx).abs());
    println!("d/dxy:   {:.15e}", (dadxy - dxy).abs());
    println!("d/dyy:   {:.15e}", (dadyy - dyy).abs());
    println!("d/dxxx:  {:.15e}", (dadxxx - dxxx).abs());
    println!();
}

// Exercise 2.1.2: indefinite integral
fn ex2_1_2() {
    let x = da!(1);
    let func = (1.0 / (1.0 + x.sqr())).integ(1); // DA integral
    let integral = x.atan(); // analytical integral DA expanded
    println!("Exercise 2.1.2: Indefinite integral\n{}\n", func - integral);
}

// Exercise 2.1.3: expand the error function
fn ex2_1_3() {
    let t = da!(1);
    let erf = 2.0 / PI.sqrt() * (-t.sqr()).exp().integ(1); // error function erf(x)
    println!("Exercise 2.1.3: Error function\n{}\n", erf);
}

// Exercise 2.2.1: DA based Newton solver
fn f(x: &DA) -> DA {
    x * x.sin() + x.cos()
}

fn ex2_2_1(x0: f64) -> f64 {
    let mut x0 = x0;
    DA::push_truncation_order(1); // for this Newton solver we only need first derivatives

    let err = 1e-14;
    let x = da!(1);
    let mut func: DA;
    let mut i = 0;

    loop {
        func = f(&(x0 + &x)); // expand f around x0
        x0 -= func.cons() / func.deriv(1).cons(); // Newton step
        i += 1;
        if (func.cons().abs() <= err) || (i >= 1000) {
            break;
        }
    }

    println!("Exercise 2.2.1: DA Newton solver\n");
    println!("root x0:           {}", x0);
    println!("residue at f(x0):  {:.15e}", f(&x0.into()).cons().abs());
    println!("Newton iterations: {}", i);

    DA::pop_truncation_order(); // don't forget to reset computation order to old value for following computations
    x0
}

// Exercise 2.2.2: expand the error function around x0
fn ex2_2_2(x0: f64) {
    let t = x0 + da!(1);
    let erf = 2.0 / PI.sqrt() * (-t.sqr()).exp().integ(1); // error function erf(x)
    println!("Exercise 2.2.2: Shifted indefinite integral\n{}\n", erf);
}

fn main() {
    DA::init(30, 2);

    ex2_1_1();
    ex2_1_2();
    ex2_1_3();

    ex2_2_1(3.6);
    ex2_2_2(1.0);
}
