//! Rust wrapper of DACE, the Differential Algebra Computational Toolbox.
//!
//! ## Introduction
//!
//! DACE-RS is a Rust wrapper of DACE, the Differential Algebra Computational Toolbox
//! (<https://github.com/dacelib/dace>).
//!
//! You can find further details on Differential Algebra and its applications
//! on that web page.
//!
//! ## Installation
//!
//! DACE-RS can be used in your project by including it as a Cargo dependency.
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! dace = "0.2"
//! ```
//!
//! If you need to use the `AlgebraicVector<DA>::invert()` function,
//! you need to include `ndarray-linalg` and specify a LAPACK binding
//! (see: <https://github.com/rust-ndarray/ndarray-linalg>).
//! 
//! Example:
//! 
//! ```toml
//! [dependencies]
//! dace = "0.2"
//! ndarray-linalg = { version = "0.16", features = ["openblas-static"] }
//! ```
//! 
//! This is needed also to run tests, e.g.:
//! `cargo test --features=ndarray-linalg/openblas-static`
//! 
//! CMake and a C compiler must be installed in the system to build the DACE Core library.
//!
//! ## Tutorials
//!
//! The original DACE C++ tutorials have been translated to Rust
//! and are available in the examples folder:
//! <https://github.com/giovannipurpura/dace-rs/tree/master/examples>
//!
//! ## Examples
//!
//! This is a quick example for basic usage:
//!
//! ```rust
//! use dace::*; // import all DACE elements
//!
//! fn main() {
//!     // initialize DACE with order 10 and 3 variables
//!     DA::init(10, 3);
//!
//!     // assign the three variables to x, y, z -- notice that integers are used here!
//!     let (x, y, z): (DA, DA, DA) = (da!(1), da!(2), da!(3));
//!     // create also some constants as DA objects -- notice that floats are used here!
//!     let (a, b, c): (DA, DA, DA) = (da!(1.0), da!(2.0), da!(3.0));
//!
//!     // compute a * sin(x) + b * cos(y) + c * tan(z)
//!     let v1: DA = &a * x.sin() + &b * y.cos() + &c * z.tan();
//!     // print the resulting DA variable
//!     println!("{v1}");
//!
//!     // do the same without using the DA constants a, b, c
//!     let v2: DA = 1.0 * x.sin() + 2.0 * y.cos() + 3.0 * z.tan();
//!     // check that we got the same result
//!     println!("v1 == v2: {}", v1 == v2);
//!
//!     // try also with AlgebraicVector<DA> and AlgebraicVector<f64>
//!     let xyz: AlgebraicVector<DA> = darray![x.sin(), y.cos(), z.tan()];
//!     let abc: AlgebraicVector<f64> = darray![1.0, 2.0, 3.0];
//!     let v3: DA = xyz.dot(&abc);
//!     // check that we got the same result
//!     println!("v1 == v3: {}", v1 == v3);
//!
//!     // try also with AlgebraicMatrix<DA> and AlgebraicMatrix<f64>
//!     let xyz: AlgebraicMatrix<DA> = darray![[x.sin(), y.cos(), z.tan()]];
//!     let abc: AlgebraicMatrix<f64> = darray![[1.0], [2.0], [3.0]];
//!     let v4: AlgebraicMatrix<DA> = xyz.dot(&abc);
//!     // check that we got the same result
//!     println!("v1 == v4: {}", v1 == v4[(0, 0)]);
//! }
//! ```

mod da;
pub mod dacecore;
mod exception;
mod interval;
mod mat;
mod vec;
pub use da::{Assign, Compile, Dot, Eval, One, Pow, Zero, DA};
pub use std::f64::consts::PI;
use std::sync::{Mutex, RwLock};

pub use interval::Interval;
pub use mat::AlgebraicMatrix;
pub use vec::{AlgebraicVector, Cross};

use exception::{check_exception, check_exception_panic, DACEException};

static DACE_MAJOR_VERSION: i32 = 2;
static DACE_MINOR_VERSION: i32 = 0;

static DACE_STRLEN: usize = 140;

static INITIALIZED: RwLock<bool> = RwLock::new(false);

/// Truncation order stack
static TO_STACK: Mutex<Vec<u32>> = Mutex::new(vec![]);

/// Create a DA object.
///
/// Alias of `DA::from(val)`
///
/// Possible arguments and outputs:
/// - u32 -> n-th DA variable
/// - f64 -> DA constant
/// - (u32, f64) -> product of n-th DA variable and constant
/// - &DA -> clone of the DA object
///
/// # Examples
///
/// ```
/// use dace::*;
///
/// // initialize DACE with order 10 and 3 variables
/// DA::init(10, 3);
///
/// let x: DA = da!(1); // first variable
/// let a: DA = da!(3.0); // constant
/// let val = &a * x.sqr(); // 3.0 * x^2
/// let der = val.deriv(1); // derive wrt 1st variable -> 6.0 * x
/// let other = 6.0 * &x; // same, in another way
/// assert_eq!(der, other); // should be equal
/// ```
#[macro_export]
macro_rules! da {
    ($val:expr) => {
        $crate::DA::from($val)
    };
}

/// Create an AlgebraicVector or an AlgebraicMatrix.
///
/// # Examples
///
/// ```
/// use dace::*;
///
/// // initialize DACE with order 10 and 3 variables
/// DA::init(10, 3);
///
/// let (x, y, z): (DA, DA, DA) = (da!(1), da!(2), da!(3));
///
/// let xyz: AlgebraicVector<DA> = darray![x.sin(), y.cos(), z.tan()];
/// let abc: AlgebraicVector<f64> = darray![1.0, 2.0, 3.0];
/// let v1: DA = xyz.dot(&abc);
///
/// let xyz: AlgebraicMatrix<DA> = darray![[x.sin(), y.cos(), z.tan()]];
/// let abc: AlgebraicMatrix<f64> = darray![[1.0], [2.0], [3.0]];
/// let v2: AlgebraicMatrix<DA> = xyz.dot(&abc);
///
/// assert_eq!(v1, v2[(0, 0)]);
/// ```
#[macro_export]
macro_rules! darray {
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {{
        $crate::AlgebraicMatrix::from(vec![$([$($x,)*],)*])
    }};
    ($($x:expr),* $(,)*) => {{
        $crate::AlgebraicVector::from(vec![$($x,)*])
    }};
}
