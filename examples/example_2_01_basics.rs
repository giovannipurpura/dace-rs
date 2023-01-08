use dace::*;
use std::panic;

// Exercise 1.1.2: first steps
fn ex1_1_2() {
    let x = da!(1);
    let func = 3.0 * (&x + 3.0) - &x + 1.0 - (&x + 8.0);
    println!("Exercise 1.1.2: First steps\n{}", func);
}

// Exercise 1.1.3: different expansion point
fn ex1_1_3() {
    let x = da!(1);
    let func = (1.0 + x).sin();
    println!("Exercise 1.1.3: Different expansion point\n{}", func);
}

// Exercise 1.1.4: a higher power
fn ex1_1_4() {
    let x = da!(1);
    let func = x.sin();
    let mut res = da!(1.0); // this makes res a constant function P(x) = 1.0
    for _ in 0..11 {
        res *= &func;
    }
    println!("Exercise 1.1.4: A higher power\n{}", res);
}

// Exercise 1.1.5: two arguments
fn ex1_1_5(x: &DA, y: &DA) -> DA {
    (1.0 + x * x + y * y).sqrt()
}

// Exercise 1.2.1: identity crisis
fn ex1_2_1() {
    let x = da!(1);
    let s2 = x.sin() * x.sin();
    let c2 = x.cos() * x.cos();
    println!(
        "Exercise 1.2.1: Identity crisis\n{}\n{}\n{}",
        s2,
        c2,
        &s2 + &c2
    );
}

// Exercise 1.2.2: Breaking bad
fn ex1_2_2(x: &DA, y: &DA) -> DA {
    let r = (x * x + y * y).sqrt();
    r.sin() / r
}

fn main() {
    DA::init(10, 2);

    let x = da!(1);
    let y = da!(2);

    ex1_1_2();
    ex1_1_3();
    ex1_1_4();

    println!("Exercise 1.1.5: Two arguments\n{}", ex1_1_5(&x, &y));

    ex1_2_1();

    println!("Exercise 1.2.2: Breaking (this is expected to panic!)\n");

    let result_1 = panic::catch_unwind(|| ex1_2_2(&DA::zero(), &DA::zero()));
    println!(
        "ex1_2_2(DA::zero(), DA::zero()) has given an error: {}\n",
        result_1.is_err()
    );

    let result_2 = panic::catch_unwind(|| ex1_2_2(&x, &y));
    println!(
        "ex1_2_2(da!(1), da!(2)) has given an error: {}\n",
        result_2.is_err()
    );
}
