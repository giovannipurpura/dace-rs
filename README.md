# DACE-RS

Rust wrapper of DACE, the Differential Algebra Computational Toolbox.

## Introduction

DACE-RS is a Rust wrapper of DACE, the Differential Algebra Computational Toolbox
(https://github.com/dacelib/dace).

You can find further details on Differential Algebra and its applications
on that web page.

## Installation

DACE-RS can be used in your project by including it as a Cargo dependency.

Add this to your `Cargo.toml`:

```toml
[build-dependencies]
dace = "0.1"
```

CMake and a C compiler must be installed in the system to build the DACE Core library.

## Tutorials

The original DACE C++ tutorials have been translated to Rust
and are available in the examples folder:
https://github.com/giovannipurpura/dace-rs/tree/master/examples

## Examples

This is a quick example for basic usage:

```rust
use dace::*; // import all DACE elements

fn main() {
    // initialize DACE with order 10 and 3 variables
    DA::init(10, 3);

    // assign the three variables to x, y, z -- notice that integers are used here!
    let (x, y, z): (DA, DA, DA) = (da!(1), da!(2), da!(3));
    // create also some constants as DA objects -- notice that floats are used here!
    let (a, b, c): (DA, DA, DA) = (da!(1.0), da!(2.0), da!(3.0));

    // compute a * sin(x) + b * cos(y) + c * tan(z)
    let v1: DA = &a * x.sin() + &b * y.cos() + &c * z.tan();
    // print the resulting DA variable
    println!("{v1}");

    // do the same without using the DA constants a, b, c
    let v2: DA = 1.0 * x.sin() + 2.0 * y.cos() + 3.0 * z.tan();
    // check that we got the same result
    println!("v1 == v2: {}", v1 == v2);

    // try also with AlgebraicVector<DA> and AlgebraicVector<f64>
    let xyz: AlgebraicVector<DA> = darray![x.sin(), y.cos(), z.tan()];
    let abc: AlgebraicVector<f64> = darray![1.0, 2.0, 3.0];
    let v3: DA = xyz.dot(&abc);
    // check that we got the same result
    println!("v1 == v3: {}", v1 == v3);

    // try also with AlgebraicMatrix<DA> and AlgebraicMatrix<f64>
    let xyz: AlgebraicMatrix<DA> = darray![[x.sin(), y.cos(), z.tan()]];
    let abc: AlgebraicMatrix<f64> = darray![[1.0], [2.0], [3.0]];
    let v4: AlgebraicMatrix<DA> = xyz.dot(&abc);
    // check that we got the same result
    println!("v1 == v4: {}", v1 == v4[(0, 0)]);
}
```
