use dace::*;
use std::fs::File;
use std::io::Write;

const VAL1: f64 = 1.685401585899429; // reference value of the integral over -1, +1
const VAL2: f64 = 1.990644530037905; // reference value of the integral over -2, +2

// Exercise 3.1.1: plot a 1D polynomial
fn ex3_1_1() {
    const X0: f64 = 0.0; // expansion point
    const N: usize = 100; // number of points in grid
    const HW: f64 = 2.0; // length of grid in each direction from expansion point

    let x = da!(1) + X0;
    let func = (-&x * &x).exp();
    let mut file = File::create("ex3_1_1.dat").unwrap();

    for i in 0..N {
        let mut xx = -HW + (i as f64) * 2.0 * HW / ((N - 1) as f64); // point on the grid on [-HW,HW]
        let rda = func.eval(xx); // note: this is not an efficient way to repeatedly evaluate the same polynomial
        xx += X0; // add expansion point x0 for double evaluation
        let rdouble = (-&xx * &xx).exp();
        writeln!(&mut file, "{xx}   {rda}   {rdouble}").unwrap();
    }

    // gnuplot command: plot 'ex3_1_1.dat'u 1:2 w l, 'ex3_1_1.dat' u 1:3 w l
    // or for the error: plot 'ex3_1_1.dat' u 1:($2-$3) w l
}

// Exercise 3.1.2: plot a 2D polynomial
fn somb_da(x: &DA, y: &DA) -> DA {
    let r = (x * x + y * y).sqrt();
    r.sin() / r
}
fn somb_f(x: f64, y: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    r.sin() / r
}

fn ex3_1_2() {
    const X0: f64 = 1.0; // expansion point x
    const Y0: f64 = 1.0; // expansion point y
    const N: usize = 50; // number of points in grid

    let x = da!(1) + X0;
    let y = da!(2) + Y0;
    let func = somb_da(&x, &y);
    let mut arg = AlgebraicVector::<f64>::zeros(2);
    let mut file = File::create("ex3_1_2.dat").unwrap();

    for i in 0..N {
        arg[0] = -1.0 + (i as f64) * 2.0 / ((N - 1) as f64); // x coordinate on the grid on [-1,1]
        for j in 0..N {
            arg[1] = -1.0 + (j as f64) * 2.0 / ((N - 1) as f64); // y coordinate on the grid on [-1,1]
            let rda = func.eval(&arg); // note: this is not an efficient way to repeatedly evaluate the same polynomial
            let rdouble = somb_f(X0 + arg[0], Y0 + arg[1]);
            writeln!(&mut file, "{}   {}   {}   {}", arg[0], arg[1], rda, rdouble).unwrap();
        }
        writeln!(&mut file).unwrap(); // empty line between lines of data for gnuplot
    }
    // gnuplot command: splot 'ex3_1_2.dat' u 1:2:3 w l, 'ex3_1_2.dat' u 1:2:4 w l
    // or for the error: splot 'ex3_1_2.dat' u 1:2:($3-$4) w l
}

// Exercise 3.1.3: Sinusitis
fn ex3_1_3() {
    DA::push_truncation_order(10);

    let x = da!(1);
    let sinda = x.sin();
    let res1 = (&x + 2.0).sin(); // compute directly sin(2+DA(1))
    let res2 = sinda.eval(&x + 2.0); // evaluate expansion of sine with 2+DA(1)
    println!("Exercise 4.1.3: Sinusitis\n{}", res1 - res2);

    DA::pop_truncation_order();
}

// Exercise 3.1.4: Gauss integral I
fn ex3_1_4() {
    let mut file = File::create("ex3_1_4.dat").unwrap();
    for order in 1..=30 {
        DA::set_truncation_order(order); // limit the computation order
        let t = da!(1);
        let erf = 2.0 / PI.sqrt() * (-t.sqr()).exp().integ(1); // error function erf(x)
        let res = erf.eval(1.0) - erf.eval(-1.0);
        writeln!(
            &mut file,
            "{}   {}   {}",
            order,
            res,
            (res - VAL1).abs().log10()
        )
        .unwrap();
    }
    // gnuplot command: plot 'ex3_1_4.dat'u 1:2 w l
    // or for the error: plot 'ex3_1_4.dat'u 1:3 w l
}

// Exercise 3.2.1 & 3.2.2: Gauss integral II
fn gauss_int(a: f64, b: f64) -> f64 // compute integral of Gaussian on interval [a,b]
{
    let t = (a + b) / 2.0 + da!(1); // expand around center point
    let func = 2.0 / PI.sqrt() * (-t.sqr()).exp().integ(1);
    func.eval((b - a) / 2.0) - func.eval(-(b - a) / 2.0) // evaluate over -+ half width
}

fn ex3_2_1() {
    const HW: f64 = 2.0; // half-width of the interval to integrate on, i.e. [-HW,HW]
    let mut file = File::create("ex3_2_1.dat").unwrap();
    let mut res: f64;

    DA::push_truncation_order(9);
    for n in 1..=30 {
        res = 0.0;
        for i in 1..=n {
            let ai = -HW + ((i - 1) as f64) * 2.0 * HW / (n as f64);
            let ai1 = -HW + (i as f64) * 2.0 * HW / (n as f64);
            res += gauss_int(ai, ai1);
        }
        writeln!(
            &mut file,
            "{}   {}   {}",
            n,
            res,
            (res - VAL2).abs().log10()
        )
        .unwrap();
    }
    DA::pop_truncation_order();
    // compare to single expansion at full computation order
    res = gauss_int(-HW, HW);
    writeln!(
        &mut file,
        "\n{}   {}   {}",
        1,
        res,
        (res - VAL2).abs().log10()
    )
    .unwrap();

    // gnuplot command: plot 'ex3_2_1.dat'u 1:3 w lp
}

fn main() {
    DA::init(30, 2); // init with maximum computation order

    ex3_1_1();
    ex3_1_2();
    ex3_1_3();
    ex3_1_4();
    ex3_2_1();
}
