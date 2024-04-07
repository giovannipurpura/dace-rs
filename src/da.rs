use crate::dacecore::*;
use crate::*;
use auto_ops::*;
pub use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::*;
pub use num_traits::{One, Pow, Zero};
use std::{
    convert::From,
    ffi::CStr,
    fmt::{Debug, Display},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops,
    os::raw::{c_char, c_uint, c_void},
    ptr::null_mut,
    str::FromStr,
    vec,
};

/// Basic DA class representing a single polynomial.
#[repr(C)]
#[derive(Debug)]
pub struct DA {
    pub(crate) len: c_uint,
    pub(crate) max: c_uint,
    pub(crate) dmonomial: *mut c_void,
}

/// compiledDA class representing a precomputed representation
/// of a polynomial for efficient evaluation.
#[derive(Debug)]
pub struct CompiledDA {
    pub(crate) dim: usize,
    pub(crate) ac: Vec<f64>,
    pub(crate) terms: u32,
    pub(crate) _vars: u32,
    pub(crate) ord: u32,
}

/// Compilation of an object into a compiledDA object.
pub trait Compile {
    /// Perform the compilation operation.
    fn compile(&self) -> CompiledDA;
}

macro_rules! simple_wrap {
    ($name_wrap:ident, $name_core:ident, $doc:literal) => {
        #[doc = $doc]
        pub fn $name_wrap(&self) -> DA {
            let mut out = Self::new();
            unsafe { $name_core(self, &mut out) };
            check_exception_panic();
            out
        }
    };
}

impl DA {
    /// Initialize the DACE control arrays and set the maximum order and the
    /// maximum number of variables.
    ///
    /// **MUST BE CALLED BEFORE ANY OTHER DA ROUTINE CAN BE USED**
    ///
    /// *This routine performs a mandatory version check to compare the version
    /// of the Rust interface used to compile the program to the version of the
    /// linked DACE library.*
    ///
    /// # Arguments
    ///
    /// * `ord` - order of the Taylor polynomials
    /// * `nvar` - number of variables considered
    ///
    /// # Panics
    ///
    /// Panics if the DACE-RS version is not compatible with the DACE Core version.
    pub fn init(ord: u32, nvar: u32) {
        if !Self::check_version() {
            panic!("Incompatible DACE core version!");
        }
        unsafe { daceInitialize(ord, nvar) };
        check_exception_panic();
        *INITIALIZED.write().unwrap() = true;
    }

    /// Initialize DACE for the current thread.
    pub fn initialize_thread() {
        unsafe { daceInitializeThread() };
        check_exception_panic();
    }

    /// Clean up DACE for the current thread.
    pub fn cleanup_thread() {
        unsafe { daceCleanupThread() };
        check_exception_panic();
    }

    /// Check if DACE has been initialized (true if initialized).
    pub fn is_initialized() -> bool {
        *INITIALIZED.read().unwrap()
    }

    /// Get DACE core version.
    ///
    /// Return format: (major, minor, patch)
    pub fn version() -> (i32, i32, i32) {
        let mut ver = (0, 0, 0);
        unsafe { daceGetVersion(&mut ver.0, &mut ver.1, &mut ver.2) };
        ver
    }

    /// Check DACE core version compatibility (true if compatible with Rust lib).
    ///
    /// To be compatible, these conditions must be satisfied:
    /// - core version major == Rust lib version major
    /// - core version minor == Rust lib version minor
    pub fn check_version() -> bool {
        let ver = Self::version();
        ver.0 == DACE_MAJOR_VERSION && ver.1 == DACE_MINOR_VERSION
    }

    /// Get the maximum order currently set for the computations
    /// (zero if undefined).
    pub fn max_order() -> u32 {
        unsafe { daceGetMaxOrder() }
    }

    /// Set the cutoff value eps to a new value and return the previous value
    /// (zero if undefined).
    pub fn set_eps(eps: f64) -> f64 {
        unsafe { daceSetEpsilon(eps) }
    }

    /// Get the cutoff value eps currently set for the computations
    /// (zero if undefined).
    pub fn eps() -> f64 {
        unsafe { daceGetEpsilon() }
    }

    /// Get the machine epsilon (experimentally determined).
    ///
    /// # Examples
    ///
    /// ```
    /// // Get machine epsilon and test that it is the smallest
    /// // number that added to 1.0 gives a value greater than 1.0
    /// use dace::DA;
    /// DA::init(2, 1);
    /// let eps = DA::eps_mac();
    /// assert_ne!(1.0 + eps, 1.0);
    /// assert_eq!(1.0 + (eps / 2.0), 1.0)
    /// ```
    pub fn eps_mac() -> f64 {
        unsafe { daceGetMachineEpsilon() }
    }

    /// Get the maximum number of variables set for the computations.
    pub fn max_variables() -> u32 {
        unsafe { daceGetMaxVariables() }
    }

    /// Get the maximum number of monomials available with the
    /// order and number of variables specified (zero if undefined).
    pub fn max_monomials() -> u32 {
        unsafe { daceGetMaxMonomials() }
    }

    /// Set the truncation order ot to a new value and return the previous value.
    ///
    /// All terms larger than the truncation order
    /// are discarded in subsequent operations.
    ///
    /// # Arguments
    ///
    /// * `ot` - new truncation order, use `None` to set to the maximum order
    pub fn set_truncation_order(ot: impl Into<Option<u32>>) -> u32 {
        let ot = ot.into().unwrap_or_else(Self::max_order);
        let prev_ot = unsafe { daceSetTruncationOrder(ot) };
        check_exception_panic();
        prev_ot
    }

    /// Get the truncation order currently set for the computations
    /// (default: max. order).
    ///
    /// All terms larger than the truncation order
    /// are discarded in subsequent operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use dace::DA;
    /// DA::init(10, 1);
    /// assert_eq!(DA::truncation_order(), 10);
    /// ```
    pub fn truncation_order() -> u32 {
        unsafe { daceGetTruncationOrder() }
    }

    /// Set a new truncation order, saving the previous one on the truncation
    /// order stack.
    ///
    /// All terms larger than the truncation order
    /// are discarded in subsequent operations.
    ///
    /// # Arguments
    ///
    /// * `ot` - new truncation order, use `None` to set to the maximum order
    ///
    /// # Examples
    ///
    /// ```
    /// use dace::DA;
    /// // start with 10
    /// DA::init(10, 1);
    /// assert_eq!(DA::truncation_order(), 10);
    /// // change to 5
    /// DA::push_truncation_order(5);
    /// assert_eq!(DA::truncation_order(), 5);
    /// // change to 2
    /// DA::push_truncation_order(2);
    /// assert_eq!(DA::truncation_order(), 2);
    /// // back to prev (5)
    /// DA::pop_truncation_order();
    /// assert_eq!(DA::truncation_order(), 5);
    /// // back to prev (10)
    /// DA::pop_truncation_order();
    /// assert_eq!(DA::truncation_order(), 10);
    /// // back to prev (no prev available: will fail and stay on 10)
    /// DA::pop_truncation_order();
    /// assert_eq!(DA::truncation_order(), 10);
    /// ```
    pub fn push_truncation_order(ot: impl Into<Option<u32>>) {
        let ot: u32 = ot.into().unwrap_or_else(Self::max_order);
        TO_STACK
            .lock()
            .unwrap()
            .push(Self::set_truncation_order(ot));
    }

    /// Restore the previous truncation order from the truncation order stack.
    ///
    /// All terms larger than the truncation order
    /// are discarded in subsequent operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use dace::DA;
    /// // start with 10
    /// DA::init(10, 1);
    /// assert_eq!(DA::truncation_order(), 10);
    /// // change to 5
    /// DA::push_truncation_order(5);
    /// assert_eq!(DA::truncation_order(), 5);
    /// // change to 2
    /// DA::push_truncation_order(2);
    /// assert_eq!(DA::truncation_order(), 2);
    /// // back to prev (5)
    /// DA::pop_truncation_order();
    /// assert_eq!(DA::truncation_order(), 5);
    /// // back to prev (10)
    /// DA::pop_truncation_order();
    /// assert_eq!(DA::truncation_order(), 10);
    /// // back to prev (no prev available: will fail and stay on 10)
    /// DA::pop_truncation_order();
    /// assert_eq!(DA::truncation_order(), 10);
    /// ```
    pub fn pop_truncation_order() -> Option<u32> {
        Some(Self::set_truncation_order(TO_STACK.lock().unwrap().pop()?))
    }

    /// Dump DACE core memory, only for debug purposes.
    pub fn memdump() {
        unsafe { daceMemoryDump() }
    }

    /// Create an empty DA object representing the constant zero function.
    pub fn new() -> Self {
        let mut da = MaybeUninit::<DA>::uninit();
        unsafe { daceAllocateDA(da.as_mut_ptr(), 0) };
        check_exception_panic();
        unsafe { da.assume_init() }
    }

    /// Create a DA object and fill it with random entries.
    ///
    /// # Arguments
    ///
    /// * `cm` - filling factor
    ///   - for `cm` < 0, the DA object is filled with random numbers
    ///   - for `cm` > 0, the DA object is filled with weighted decaying numbers
    pub fn random(cm: f64) -> Self {
        let mut out = DA::new();
        unsafe { daceCreateRandom(&mut out, cm) };
        check_exception_panic();
        out
    }

    /// Get the constant part of the DA object.
    pub fn cons(&self) -> f64 {
        let out = unsafe { daceGetConstant(self) };
        check_exception_panic();
        out
    }
    /// Get the linear part of the DA object.
    pub fn linear(&self) -> AlgebraicVector<f64> {
        let mut out = AlgebraicVector::zeros(Self::max_variables() as usize);
        unsafe { daceGetLinear(self, out.as_mut_ptr()) };
        check_exception_panic();
        out
    }

    /// Compute the gradient of a DA object.
    pub fn gradient(&self) -> AlgebraicVector<DA> {
        let nvar = Self::max_variables() as usize;
        Array::from_shape_fn((nvar,), |i| self.deriv(i as u32 + 1)).into()
    }

    /// Get a specific coefficient of a DA object.
    ///
    /// # Arguments
    ///
    /// * `jj` - vector of the exponents of the coefficient to retrieve
    ///
    /// # Example
    ///
    /// ```
    /// use dace::*;
    /// DA::init(10, 2);
    /// let (x, y) = (da!(1), da!(2));
    /// let v = (1.0 + 3.0 * &x) * (2.0 + 4.0 * &y).pow(2);
    /// let x1_y2_coeff = v.coefficient(&vec![1, 2]);
    /// assert_eq!(x1_y2_coeff, 48.0);
    /// ```
    pub fn coefficient(&self, jj: &Vec<u32>) -> f64 {
        let nvar = Self::max_variables() as usize;
        let ptr;
        let mut temp;
        if jj.len() >= nvar {
            ptr = jj.as_ptr();
        } else {
            temp = jj.clone();
            temp.resize(nvar, 0);
            ptr = temp.as_ptr();
        }
        let coeff = unsafe { daceGetCoefficient(self, ptr) };
        check_exception_panic();
        coeff
    }

    /// Set a specific coefficient into a DA object.
    ///
    /// # Arguments
    ///
    /// * `jj` - vector of the exponents of the coefficient to set
    /// * `coeff` - value to be set as coefficient
    ///
    /// # Example
    ///
    /// ```
    /// use dace::*;
    /// DA::init(10, 2);
    /// let (x, y) = (da!(1), da!(2));
    /// let mut v = DA::new();
    /// v.set_coefficient(&vec![1, 2], 12.0);
    /// assert_eq!(v, 12.0 * x * y.pow(2));
    /// ```
    pub fn set_coefficient(&mut self, jj: &Vec<u32>, coeff: f64) {
        let nvar = Self::max_variables() as usize;
        let ptr;
        let mut temp;
        if jj.len() >= nvar {
            ptr = jj.as_ptr();
        } else {
            temp = jj.clone();
            temp.resize(nvar, 0);
            ptr = temp.as_ptr();
        }
        unsafe { daceSetCoefficient(self, ptr, coeff) };
        check_exception_panic();
    }

    /// Multiply the DA object with another DA object monomial by monomial.
    ///
    /// This is the equivalent of coefficient-wise multiplication (like in DA addition).
    ///
    /// # Arguments
    ///
    /// * `oth` - DA object to multiply with coefficient-wise
    pub fn multiply_monomials<T: AsRef<Self>>(&self, oth: T) -> Self {
        let mut out = Self::new();
        unsafe { daceMultiplyMonomials(self, oth.as_ref(), &mut out) };
        check_exception_panic();
        out
    }

    /// Divide by independent variable `var` raised to power `p`.
    ///
    /// # Arguments
    ///
    /// * `var` - independent variable number to divide by
    /// * `p` - power of the independent variable
    pub fn divide(&self, var: u32, p: u32) -> Self {
        let mut out = Self::new();
        unsafe { daceDivideByVariable(self, var, p, &mut out) };
        check_exception_panic();
        out
    }

    /// Compute the derivative of a DA object with respect to variable `i`.
    ///
    /// # Arguments
    ///
    /// * `i` - variable with respect to which the derivative is calculated
    pub fn deriv(&self, i: u32) -> Self {
        let mut out = Self::new();
        unsafe { daceDifferentiate(i, self, &mut out) };
        check_exception_panic();
        out
    }

    /// Compute the integral of a DA object with respect to variable `i`.
    ///
    /// # Arguments
    ///
    /// * `i` - variable with respect to which the integral is calculated
    pub fn integ(&self, i: u32) -> Self {
        let mut out = Self::new();
        unsafe { daceIntegrate(i, self, &mut out) };
        check_exception_panic();
        out
    }

    /// Returns a DA object with all monomials of order
    /// less than `min` and greater than `max` removed.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum order to keep in the DA object
    /// * `max` - The maximum order to keep in the DA object (or `None`)
    pub fn trim(&self, imin: u32, imax: impl Into<Option<u32>>) -> Self {
        let mut out = Self::new();
        let imax = imax.into().unwrap_or_else(Self::max_order);
        unsafe { daceTrim(self, imin, imax, &mut out) };
        check_exception_panic();
        out
    }

    simple_wrap!{trunc, daceTruncate, "Truncate the constant part of a DA object to an integer."}

    simple_wrap!{round, daceRound, "Round the constant part of a DA object to an integer."}

    /// Compute the p-th root of a DA object.
    ///
    /// # Arguments
    ///
    /// * `p` - root to be computed
    pub fn root(&self, p: i32) -> DA {
        let mut out = DA::new();
        unsafe { daceRoot(self, p, &mut out) };
        check_exception_panic();
        out
    }

    simple_wrap!{sqr, daceSquare, "Compute the square of a DA object."}

    /// Compute the square of a DA object (alias of `sqr`).
    #[inline(always)]
    pub fn square(&self) -> DA {
        self.sqr()
    }

    simple_wrap!{sqrt, daceSquareRoot, "Compute the square root of a DA object."}

    simple_wrap!{isrt, daceInverseSquareRoot, "Compute the inverse square root of a DA object."}

    simple_wrap!{cbrt, daceCubicRoot, "Compute the cubic root of a DA object."}

    simple_wrap!{icrt, daceInverseCubicRoot, "Compute the inverse cubic root of a DA object."}

    /// Compute the hypotenuse (`(a.sqr() + b.sqr()).sqrt()`) of a DA object and the given DA argument.
    pub fn hypot<T: AsRef<DA>>(&self, oth: T) -> DA {
        let mut out = DA::new();
        unsafe { daceHypotenuse(self, oth.as_ref(), &mut out) };
        check_exception_panic();
        out
    }

    simple_wrap!{exp, daceExponential, "Compute the exponential of a DA object."}

    simple_wrap!{log, daceLogarithm, "Compute the natural logarithm of a DA object."}

    /// Compute the logarithm of a DA object with respect to a given base.
    ///
    /// # Arguments
    ///
    /// * `b` - logarithm base
    pub fn logb(&self, b: f64) -> DA {
        let mut out = DA::new();
        unsafe { daceLogarithmBase(self, b, &mut out) };
        check_exception_panic();
        out
    }

    simple_wrap!{log10, daceLogarithm10, "Compute the 10 based logarithm of a DA object."}

    simple_wrap!{log2, daceLogarithm2, "Compute the 2 based logarithm of a DA object."}

    simple_wrap!{sin, daceSine, "Compute the sine of a DA object."}

    simple_wrap!{cos, daceCosine, "Compute the cosine of a DA object."}

    simple_wrap!{tan, daceTangent, "Compute the tangent of a DA object."}

    simple_wrap!{asin, daceArcSine, "Compute the arcsine of a DA object."}

    /// Compute the arccosine of a DA object.
    #[inline(always)]
    pub fn arcsin(&self) -> Self {
        self.asin()
    }

    simple_wrap!{acos, daceArcCosine, "Compute the arccosine of a DA object."}

    /// Compute the arccosine of a DA object.
    #[inline(always)]
    pub fn arccos(&self) -> Self {
        self.acos()
    }

    simple_wrap!{atan, daceArcTangent, "Compute the arctangent of a DA object."}

    /// Compute the arctangent of a DA object.
    #[inline(always)]
    pub fn arctan(&self) -> Self {
        self.atan()
    }

    /// Compute the four-quadrant arctangent of Y/X.
    /// Y is the current DA object, whereas X is the given DA.
    ///
    /// # Arguments
    ///
    /// * `oth` - X
    pub fn atan2<T: AsRef<Self>>(&self, oth: T) -> Self {
        let mut out = Self::new();
        unsafe { daceArcTangent2(self, oth.as_ref(), &mut out) };
        check_exception_panic();
        out
    }

    /// Compute the four-quadrant arctangent of Y/X.
    /// Y is the current DA object, whereas X is the given DA.
    ///
    /// # Arguments
    ///
    /// * `oth` - X
    #[inline(always)]
    pub fn arctan2<T: AsRef<Self>>(&self, oth: T) -> Self {
        self.atan2(oth)
    }

    simple_wrap!{sinh, daceHyperbolicSine, "Compute the hyperbolic sine of a DA object."}

    simple_wrap!{cosh, daceHyperbolicCosine, "Compute the hyperbolic cosine of a DA object."}

    simple_wrap!{tanh, daceHyperbolicTangent, "Compute the hyperbolic tangent of a DA object."}

    simple_wrap!{asinh, daceHyperbolicArcSine, "Compute the hyperbolic arcsine of a DA object."}

    /// Compute the hyperbolic arcsine of a DA object.
    #[inline(always)]
    pub fn arcsinh(&self) -> Self {
        self.asinh()
    }

    simple_wrap!{acosh, daceHyperbolicArcCosine, "Compute the hyperbolic arccosine of a DA object."}

    /// Compute the hyperbolic arccosine of a DA object.
    #[inline(always)]
    pub fn arccosh(&self) -> Self {
        self.acosh()
    }

    simple_wrap!{atanh, daceHyperbolicArcTangent, "Compute the hyperbolic arctangent of a DA object."}

    /// Compute the hyperbolic arctangent of a DA object.
    #[inline(always)]
    pub fn arctanh(&self) -> Self {
        self.atanh()
    }

    simple_wrap!{minv, daceMultiplicativeInverse, "Compute the multiplicative inverse of a DA object."}

    simple_wrap!{erf, daceErrorFunction, "Compute the error function of a DA object."}

    simple_wrap!{erfc, daceComplementaryErrorFunction, "Compute the complementary error function of a DA object."}

    /// Compute the `n`-th Bessel function of first type `J_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `n` - order of the Bessel function
    pub fn besselj(&self, n: i32) -> Self {
        let mut out = Self::new();
        unsafe { daceBesselJFunction(self, n, &mut out) };
        check_exception_panic();
        out
    }

    /// Compute the `n`-th Bessel function of second type `Y_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `n` - order of the Bessel function
    pub fn bessely(&self, n: i32) -> Self {
        let mut out = Self::new();
        unsafe { daceBesselYFunction(self, n, &mut out) };
        check_exception_panic();
        out
    }

    /// Compute the `n`-th modified Bessel function of first type `I_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `n` - order of the Bessel function
    /// * `scaled` if true, the modified Bessel function is scaled
    ///    by a factor `exp(-x)`, i.e. `exp(-x)I_n(x)` is returned.
    pub fn besseli(&self, n: i32, scaled: bool) -> Self {
        let mut out = Self::new();
        unsafe { daceBesselIFunction(self, n, scaled, &mut out) };
        check_exception_panic();
        out
    }

    /// Compute the `n`-th modified Bessel function of second type `K_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `n` - order of the Bessel function
    /// * `scaled` if true, the modified Bessel function is scaled
    ///    by a factor `exp(-x)`, i.e. `exp(-x)K_n(x)` is returned.
    pub fn besselk(&self, n: i32, scaled: bool) -> Self {
        let mut out = Self::new();
        unsafe { daceBesselKFunction(self, n, scaled, &mut out) };
        check_exception_panic();
        out
    }

    simple_wrap!{gamma, daceGammaFunction, "Compute the Gamma function of a DA object."}

    simple_wrap!{loggamma, daceLogGammaFunction, "Compute the Logarithmic Gamma function (i.e. the natural logarithm of Gamma) of a DA object."}

    /// Compute the `n`-th order Psi function, i.e. the (`n`+1)st derivative of the Logarithmic Gamma function, of a DA object.
    pub fn psi(&self, n: u32) -> Self {
        let mut out = Self::new();
        unsafe { dacePsiFunction(self, n, &mut out) };
        check_exception_panic();
        out
    }

    /// Get the number of non-zero coefficients of a DA object.
    #[inline]
    pub fn size(&self) -> u32 {
        self.len
    }

    /// Compute the max norm of a DA object.
    pub fn abs(&self) -> f64 {
        let out = unsafe { daceAbsoluteValue(self) };
        check_exception_panic();
        out
    }

    /// Compute different types of norms for a DA object.
    ///
    /// # Arguments
    ///
    /// * `type_` type of norm to be computed. Possible norms are:
    ///   - 0: Max norm
    ///   - 1: Sum norm
    ///   - oth: Vector norm of given type
    pub fn norm(&self, type_: u32) -> f64 {
        let out = unsafe { daceNorm(self, type_) };
        check_exception_panic();
        out
    }

    /// Extract different types of order sorted norms from a DA object.
    ///
    /// # Arguments
    ///
    /// * `var` order
    ///   - 0: Terms are sorted by their order
    ///   - oth: Terms are sorted by the exponent of variable `var`
    /// * `type_` type of norm to be computed. Possible norms are:
    ///   - 0: Max norm
    ///   - 1: Sum norm
    ///   - oth: Vector norm of given type
    pub fn order_norm(&self, var: u32, type_: u32) -> AlgebraicVector<f64> {
        let mut v = AlgebraicVector::<f64>::zeros(Self::max_order() as usize + 1);
        unsafe { daceOrderedNorm(self, var, type_, v.as_mut_ptr()) };
        check_exception_panic();
        v
    }

    /// Estimate different types of order sorted norms for terms
    /// of a DA object up to a specified order.
    ///
    /// # Arguments
    ///
    /// * `var` - order
    ///   - 0: Terms are sorted by their order
    ///   - oth: Terms are sorted by the exponent of variable `var`
    /// * `type_` - type of norm to be computed. Possible norms are:
    ///   - 0: Max norm
    ///   - 1: Sum norm
    ///   - oth: Vector norm of given type
    /// * `nc` - maximum order to be estimated
    pub fn estim_norm(&self, var: u32, type_: u32, nc: u32) -> AlgebraicVector<f64> {
        let mut v = AlgebraicVector::<f64>::zeros(nc as usize + 1);
        unsafe { daceEstimate(self, var, type_, v.as_mut_ptr(), null_mut(), nc) };
        check_exception_panic();
        v
    }

    /// Estimate different types of order sorted norms for terms
    /// of a DA object up to a specified order with error estimates.
    ///
    /// # Arguments
    ///
    /// * `var` - order
    ///   - 0: Terms are sorted by their order
    ///   - oth: Terms are sorted by the exponent of variable `var`
    /// * `type_` - type of norm to be computed. Possible norms are:
    ///   - 0: Max norm
    ///   - 1: Sum norm
    ///   - oth: Vector norm of given type
    /// * `nc` - maximum order to be estimated
    pub fn estim_norm_err(&self, var: u32, type_: u32, nc: u32) -> AlgebraicVector<f64> {
        let mut err = AlgebraicVector::<f64>::zeros(nc.min(Self::max_order()) as usize + 1);
        let mut v = AlgebraicVector::<f64>::zeros(nc as usize + 1);
        unsafe { daceEstimate(self, var, type_, v.as_mut_ptr(), err.as_mut_ptr(), nc) };
        check_exception_panic();
        v
    }

    /// Compute lower and upper bounds of a DA object.
    pub fn bound(&self) -> Interval {
        let mut out = Interval {
            m_lb: 0.0,
            m_ub: 0.0,
        };
        unsafe { daceGetBounds(self, &mut out.m_lb, &mut out.m_ub) };
        check_exception_panic();
        out
    }

    /// Estimate the convergence radius of the DA object.
    ///
    /// # Arguments
    ///
    /// * `eps` - requested tolerance
    /// * `type_` - type of norm (sum norm is used as default)
    pub fn conv_radius(&self, eps: f64, type_: u32) -> f64 {
        let ord = Self::truncation_order();
        let res = self.estim_norm(0, type_, ord + 1);
        (eps / res[ord as usize + 1]).powf(1.0 / (ord + 1) as f64)
    }

    /// Partial evaluation of a DA object.
    ///
    /// In the DA object, variable `var` is replaced by the value `val`.
    ///
    /// # Arguments
    ///
    /// * `var` - variable number to be replaced
    /// * `val` - value by which to replace the variable
    pub fn plug(&self, var: u32, val: f64) -> DA {
        let mut out = DA::new();
        unsafe { daceEvalVariable(self, var, val, &mut out) };
        check_exception_panic();
        out
    }

    /// Evaluates the DA vector using the coefficients in argument `values`
    /// as the values for each monomial.
    ///
    /// This is equivalent to a monomial-wise dot product of two DA vectors.
    ///
    /// # Arguments
    ///
    /// * `values` - DA vector containing the values of each monomial
    pub fn eval_monomials<T: AsRef<DA>>(&self, values: T) -> f64 {
        let out = unsafe { daceEvalMonomials(self, values.as_ref()) };
        check_exception_panic();
        out
    }

    /// Partial evaluation of a DA object.
    ///
    /// In the DA object, variable `from` is replaced by the value `val` times variable `to`.
    ///
    /// # Arguments
    ///
    /// * `from` - variable number to be replaced
    /// * `to` - variable number to be inserted instead
    /// * `val` - value by which to scale the inserted variable
    pub fn replace_variable(&self, from: u32, to: u32, val: f64) -> DA {
        let mut out = DA::new();
        unsafe { daceReplaceVariable(self, from, to, val, &mut out) };
        check_exception_panic();
        out
    }

    /// Scaling of an independent variable.
    ///
    /// In the DA object, variable `var` is replaced by the value `val` times `var`.
    ///
    /// # Arguments
    ///
    /// * `var` - variable number to be scaled
    /// * `val` - value by which to scale the variable
    pub fn scale_variable(&self, var: u32, val: f64) -> DA {
        let mut out = DA::new();
        unsafe { daceScaleVariable(self, var, val, &mut out) };
        check_exception_panic();
        out
    }

    /// Affine translation of an independent variable.
    ///
    /// In the DA object, variable `var` is replaced by `a * var + c`.
    ///
    /// # Arguments
    ///
    /// * `var` - variable number to be translated
    /// * `a` - value by which to scale the variable
    /// * `c` - value by which to shift the variable
    pub fn translate_variable(&self, var: u32, a: f64, c: f64) -> DA {
        let mut out = DA::new();
        unsafe { daceTranslateVariable(self, var, a, c, &mut out) };
        check_exception_panic();
        out
    }

    /// Check if the DA object has only a constant part.
    pub fn is_constant(&self) -> bool {
        *self == self.cons()
    }

    /// Print the content of the DA object.
    pub fn print(&self) {
        unsafe { dacePrint(self) };
    }

    /// Convert a DA object to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut size = 0;
        unsafe { daceExportBlob(self, null_mut(), &mut size) };
        let mut out = vec![0; size as usize];
        unsafe { daceExportBlob(self, out.as_mut_ptr().cast(), &mut size) };
        check_exception_panic();
        out
    }
}

impl AsRef<DA> for DA {
    #[inline]
    fn as_ref(&self) -> &DA {
        self
    }
}

impl AsMut<DA> for DA {
    #[inline]
    fn as_mut(&mut self) -> &mut DA {
        self
    }
}

/// Assignment of a value to an object.
pub trait Assign<T> {
    /// Perform the assignment operation.
    fn assign(&mut self, oth: T);
}

impl Assign<(u32, f64)> for DA {
    /// Assign a value to the DA variable, as `c` times the `i`-th variable.
    ///
    /// # Arguments
    ///
    /// * `i` - variable number
    /// * `c` - multiplicative constant
    ///
    /// # Example
    ///
    /// ```
    /// use dace::*;
    /// DA::init(3, 2);
    /// // create a var using the `assign` method
    /// let mut var_1 = DA::new();
    /// var_1.assign((2, 3.0));
    /// // create the same value in another way
    /// let var_2 = da!(2) * 3.0;
    /// // check that they are equal
    /// assert_eq!(var_1, var_2);
    /// ```
    fn assign(&mut self, oth: (u32, f64)) {
        unsafe { daceCreateVariable(self, oth.0, oth.1) };
        check_exception_panic();
    }
}

impl Assign<u32> for DA {
    /// Assign the value of the n-th variable to this DA variable.
    ///
    /// # Arguments
    ///
    /// * `oth` - variable number to be assigned
    fn assign(&mut self, oth: u32) {
        unsafe { daceCreateVariable(self, oth, 1.0) };
        check_exception_panic();
    }
}

impl Assign<f64> for DA {
    /// Assign the value of a f64 constant to this DA variable.
    ///
    /// # Arguments
    ///
    /// * `oth` - f64 to be assigned
    fn assign(&mut self, oth: f64) {
        unsafe { daceCreateConstant(self, oth) };
        check_exception_panic();
    }
}

impl Assign<&DA> for DA {
    /// Assign the value of another DA variable to this DA variable.
    ///
    /// # Arguments
    ///
    /// * `oth` - DA variable to copy from
    ///
    /// # Example
    ///
    /// ```
    /// use dace::*;
    /// DA::init(3, 2);
    /// // create a value in another way
    /// let var_1 = da!(2) * 3.0;
    /// // create a var using the `assign` method
    /// let mut var_2 = DA::new();
    /// var_2.assign(&var_1);
    /// // check that they are equal
    /// assert_eq!(var_1, var_2);
    /// ```
    fn assign(&mut self, oth: &DA) {
        unsafe { daceCopy(oth, self) };
        check_exception_panic();
    }
}

impl Assign<DA> for DA {
    /// Assign the value of another DA variable to this DA variable.
    ///
    /// # Arguments
    ///
    /// * `val` - DA variable to copy from
    ///
    /// # Example
    ///
    /// ```
    /// use dace::*;
    /// DA::init(3, 2);
    /// // create a value in another way
    /// let var_1 = da!(2) * 3.0;
    /// let var_1_clone = var_1.clone();
    /// // create a var using the `assign` method
    /// let mut var_2 = DA::new();
    /// var_2.assign(var_1);
    /// // check that they are equal
    /// assert_eq!(var_1_clone, var_2);
    /// ```
    fn assign(&mut self, mut oth: DA) {
        *self = Self {
            len: oth.len,
            max: oth.max,
            dmonomial: oth.dmonomial,
        };
        unsafe { daceInvalidateDA(&mut oth) };
        check_exception_panic();
    }
}

impl PartialEq<f64> for DA {
    fn eq(&self, other: &f64) -> bool {
        (self - *other).size() == 0
    }
}

impl PartialEq for DA {
    fn eq(&self, other: &Self) -> bool {
        (self - other).size() == 0
    }
}

impl Eq for DA {}

impl Display for DA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut nstr = unsafe { daceGetMaxMonomials() } + 2;
        let mut vec = vec![0 as c_char; nstr as usize * DACE_STRLEN];
        let vec_ptr = vec.as_mut_ptr();
        unsafe { daceWrite(self, vec_ptr, &mut nstr) };
        for i in 0..nstr {
            let ptr = unsafe { vec_ptr.offset((i as usize * DACE_STRLEN) as isize) };
            let s = unsafe { CStr::from_ptr(ptr) };
            let s = s.to_str().expect("Error decoding string");
            writeln!(f, "{s}")?;
        }
        check_exception_panic();
        Ok(())
    }
}

impl ScalarOperand for DA {}
impl ScalarOperand for &'static DA {}

// Add

impl_op_ex!(+ |a: &DA, b: &DA| -> DA {
    let mut c = DA::new();
    unsafe { daceAdd(a, b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(+= |a: &mut DA, b: &DA| {
    unsafe { daceAdd(a, b, a) };
    check_exception_panic();
});

impl_op_ex_commutative!(+ |a: &DA, b: &f64| -> DA {
    let mut c = DA::new();
    unsafe { daceAddDouble(a, *b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(+= |a: &mut DA, b: &f64| {
    unsafe { daceAddDouble(a, *b, a) };
    check_exception_panic();
});

// Sub

impl_op_ex!(-|a: &DA, b: &DA| -> DA {
    let mut c = DA::new();
    unsafe { daceSubtract(a, b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(-= |a: &mut DA, b: &DA| {
    unsafe { daceSubtract(a, b, a) };
    check_exception_panic();
});

impl_op_ex!(-|a: &DA, b: &f64| -> DA {
    let mut c = DA::new();
    unsafe { daceSubtractDouble(a, *b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(-|a: &f64, b: &DA| -> DA {
    let mut c = DA::new();
    unsafe { daceDoubleSubtract(b, *a, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(-= |a: &mut DA, b: &f64| {
    unsafe { daceSubtractDouble(a, *b, a) };
    check_exception_panic();
});

// Mul

impl_op_ex!(*|a: &DA, b: &DA| -> DA {
    let mut c = DA::new();
    unsafe { daceMultiply(a, b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(*= |a: &mut DA, b: &DA| {
    unsafe { daceMultiply(a, b, a) };
    check_exception_panic();
});

impl_op_ex_commutative!(*|a: &DA, b: &f64| -> DA {
    let mut c = DA::new();
    unsafe { daceMultiplyDouble(a, *b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(*= |a: &mut DA, b: &f64| {
    unsafe { daceMultiplyDouble(a, *b, a) };
    check_exception_panic();
});

// Div

impl_op_ex!(/ |a: &DA, b: &DA| -> DA {
    let mut c = DA::new();
    unsafe { daceDivide(a, b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(/= |a: &mut DA, b: &DA| {
    unsafe { daceDivide(a, b, a) };
    check_exception_panic();
});

impl_op_ex!(/ |a: &DA, b: &f64| -> DA {
    let mut c = DA::new();
    unsafe { daceDivideDouble(a, *b, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(/ |a: f64, b: &DA| -> DA {
    let mut c = DA::new();
    unsafe { daceDoubleDivide(b, a, &mut c) };
    check_exception_panic();
    c
});

impl_op_ex!(/= |a: &mut DA, b: &f64| {
    unsafe { daceDivideDouble(a, *b, a) };
    check_exception_panic();
});

// Neg

impl ops::Neg for DA {
    type Output = DA;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl ops::Neg for &DA {
    type Output = DA;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

// Mod

impl ops::Rem<f64> for DA {
    type Output = DA;
    /// Compute the floating-point remainder of `c`/`p` (`c` modulo `p`),
    /// where `c` is the constant part of the current DA object.
    ///
    /// # Arguments
    ///
    /// * `p` - costant with respect to which the modulo function is computed.
    fn rem(self, rhs: f64) -> Self::Output {
        &self % rhs
    }
}

impl ops::Rem<f64> for &DA {
    type Output = DA;
    /// Compute the floating-point remainder of `c`/`p` (`c` modulo `p`),
    /// where `c` is the constant part of the current DA object.
    ///
    /// # Arguments
    ///
    /// * `p` - costant with respect to which the modulo function is computed.
    fn rem(self, rhs: f64) -> Self::Output {
        let mut out = DA::new();
        unsafe { daceModulo(self, rhs, &mut out) };
        check_exception_panic();
        out
    }
}

impl ops::RemAssign<f64> for DA {
    /// Compute the floating-point remainder of `c`/`p` (`c` modulo `p`),
    /// where `c` is the constant part of the current DA object.
    ///
    /// # Arguments
    ///
    /// * `p` - costant with respect to which the modulo function is computed.
    fn rem_assign(&mut self, rhs: f64) {
        unsafe { daceModulo(self, rhs, self) };
        check_exception_panic();
    }
}

impl Pow<i32> for &DA {
    type Output = DA;
    /// Elevate a DA object to a given integer power.
    ///
    /// # Arguments
    ///
    /// * `p` - power to which the DA object is raised
    fn pow(self: Self, p: i32) -> DA {
        let mut out = DA::new();
        unsafe { dacePower(self, p, &mut out) };
        check_exception_panic();
        out
    }
}

impl Pow<f64> for &DA {
    type Output = DA;
    /// Elevate a DA object to a given real power.
    ///
    /// # Arguments
    ///
    /// * `p` - power to which the DA object is raised
    fn pow(self: Self, p: f64) -> DA {
        let mut out = DA::new();
        unsafe { dacePowerDouble(self, p, &mut out) };
        check_exception_panic();
        out
    }
}

impl<T: AsRef<DA>> Pow<T> for &DA {
    type Output = DA;
    /// Elevate a DA object to a given DA power.
    ///
    /// # Arguments
    ///
    /// * `p` - power to which the DA object is raised
    fn pow(self: Self, p: T) -> DA {
        let p_ref = p.as_ref();
        // avoid using log formula if p is constant
        if p_ref.is_constant() {
            return self.pow(p_ref.cons());
        }
        let mut out = DA::new();
        unsafe {
            daceLogarithm(self, &mut out);
            daceMultiply(p_ref, &out, &mut out);
            daceExponential(&out, &mut out);
        }
        check_exception_panic();
        out
    }
}

impl From<(u32, f64)> for DA {
    fn from(ic: (u32, f64)) -> Self {
        let mut out = Self::new();
        out.assign(ic);
        out
    }
}

impl From<u32> for DA {
    fn from(i: u32) -> Self {
        let mut out = Self::new();
        out.assign(i);
        out
    }
}

impl From<f64> for DA {
    fn from(c: f64) -> Self {
        let mut out = Self::new();
        out.assign(c);
        out
    }
}

impl From<&DA> for DA {
    fn from(oth: &DA) -> Self {
        oth.to_owned()
    }
}

impl TryFrom<&Vec<u8>> for DA {
    type Error = DACEException;

    /// Convert bytes to DA objects.
    ///
    /// # Example
    ///
    /// ```
    /// use dace::*;
    /// DA::init(10, 2);
    /// let x = da!(1) * 2.0 + da!(2).pow(2);
    /// let bytes = x.to_bytes();
    /// let x_decoded = DA::try_from(&bytes).unwrap();
    /// assert_eq!(x, x_decoded);
    /// ```
    fn try_from(blob: &Vec<u8>) -> Result<Self, DACEException> {
        let mut out = Self::new();
        unsafe { daceImportBlob(blob.as_ptr().cast(), &mut out) };
        check_exception()?;
        Ok(out)
    }
}

impl FromStr for DA {
    type Err = DACEException;

    /// Convert a string to a DA object.
    fn from_str(s: &str) -> Result<Self, DACEException> {
        let mut out = Self::new();
        let nstr = s.lines().count();
        let mut vec = vec![0 as c_char; nstr * DACE_STRLEN];
        let mut vec_ptr = vec.as_mut_ptr();
        unsafe {
            for line in s.lines() {
                vec_ptr.copy_from(line.as_ptr() as *const c_char, line.len());
                vec_ptr = vec_ptr.offset(DACE_STRLEN as isize);
            }
            daceRead(&mut out, vec.as_ptr(), nstr as c_uint);
        }
        check_exception()?;
        Ok(out)
    }
}

impl Clone for DA {
    fn clone(&self) -> Self {
        let mut out = Self::new();
        unsafe { daceCopy(self, &mut out) };
        check_exception_panic();
        out
    }
}

impl Hash for DA {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_bytes().hash(state);
    }
}

impl Drop for DA {
    /// Destroy a DA object and free the associated object in the DACE core.
    fn drop(&mut self) {
        unsafe {
            daceFreeDA(self);
            if daceGetError() != 0 {
                daceClearError();
            }
        }
    }
}

impl Compile for DA {
    // Compile current DA object and create a compiledDA object.
    fn compile(&self) -> CompiledDA {
        darray![self.to_owned()].compile()
    }
}

impl Default for DA {
    /// Get a zero-valued constant DA object.
    fn default() -> Self {
        Self::new()
    }
}

impl Zero for DA {
    fn zero() -> Self {
        DA::new()
    }
    fn is_zero(&self) -> bool {
        self.size() == 0
    }
    fn set_zero(&mut self) {
        self.assign(0.0)
    }
}

impl One for DA {
    fn one() -> Self {
        da!(1.0)
    }
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self == 1.0
    }
    fn set_one(&mut self) {
        self.assign(1.0);
    }
}

/// Evaluation of a DA object with another object.
pub trait Eval<Args> {
    /// The resulting type after applying the evaluation operation.
    type Output;
    /// Perform the evaluation operation.
    fn eval(&self, args: Args) -> Self::Output;
}

impl Eval<&AlgebraicVector<f64>> for CompiledDA {
    type Output = AlgebraicVector<f64>;
    fn eval(&self, args: &AlgebraicVector<f64>) -> Self::Output {
        let mut res = AlgebraicVector::<f64>::zeros(self.dim);
        let narg = args.len();
        let mut xm = AlgebraicVector::<f64>::ones(self.ord as usize + 1);

        // constant part
        for i in 0..self.dim {
            res[i] = self.ac[i + 2];
        }

        // higher order terms
        let mut p = 2 + self.dim as usize;
        for _ in 1..self.terms {
            let jl = self.ac[p] as usize;
            p += 1;
            let jv = self.ac[p] as usize - 1;
            p += 1;
            if jv < narg {
                xm[jl] = xm[jl - 1] * args[jv];
            } else {
                xm[jl] = 0.0;
            }
            for j in 0..self.dim {
                res[j] += xm[jl] * self.ac[p];
                p += 1;
            }
        }
        res
    }
}

impl Eval<&AlgebraicVector<DA>> for CompiledDA {
    type Output = AlgebraicVector<DA>;
    fn eval(&self, args: &AlgebraicVector<DA>) -> Self::Output {
        let mut res = AlgebraicVector::<DA>::zeros(self.dim);
        let narg = args.len();
        let mut jlskip = self.ord as usize + 1;
        let mut p: usize = 2;
        let mut xm = AlgebraicVector::<DA>::ones(self.ord as usize + 1);
        let mut tmp = DA::new();

        // constant part
        for i in 0..self.dim {
            res[i].assign(self.ac[p]);
            p += 1;
        }

        // higher order terms
        for _ in 1..self.terms {
            let jl = self.ac[p] as usize;
            p += 1;
            let jv = self.ac[p] as usize - 1;
            p += 1;
            if jl > jlskip {
                p += self.dim;
                continue;
            }
            if jv >= narg {
                jlskip = jl;
                p += self.dim;
                continue;
            }
            jlskip = self.ord as usize + 1;
            unsafe { daceMultiply(&xm[jl - 1], &args[jv], &mut xm[jl]) };
            for j in 0..self.dim {
                if self.ac[p] != 0.0 {
                    unsafe { daceMultiplyDouble(&xm[jl], self.ac[p], &mut tmp) };
                    res[j] += &tmp;
                }
                p += 1;
            }
        }

        check_exception_panic();
        res
    }
}

impl Eval<&AlgebraicVector<f64>> for DA {
    type Output = f64;
    fn eval(&self, args: &AlgebraicVector<f64>) -> Self::Output {
        self.compile().eval(args)[0]
    }
}

impl Eval<&AlgebraicVector<f64>> for AlgebraicVector<DA> {
    type Output = AlgebraicVector<f64>;
    fn eval(&self, args: &AlgebraicVector<f64>) -> Self::Output {
        self.compile().eval(args)
    }
}

impl Eval<&AlgebraicVector<DA>> for AlgebraicVector<DA> {
    type Output = AlgebraicVector<DA>;
    fn eval(&self, args: &AlgebraicVector<DA>) -> Self::Output {
        self.compile().eval(args)
    }
}

impl Eval<&AlgebraicVector<DA>> for DA {
    type Output = DA;
    fn eval(&self, args: &AlgebraicVector<DA>) -> DA {
        let x = self.compile();
        CompiledDA::eval(&x, args)[0].to_owned()
    }
}

impl Eval<f64> for CompiledDA {
    type Output = AlgebraicVector<f64>;
    fn eval(&self, arg: f64) -> Self::Output {
        self.eval(&darray![arg]).into()
    }
}

impl<T: AsRef<DA>> Eval<T> for CompiledDA {
    type Output = AlgebraicVector<DA>;
    fn eval(&self, arg: T) -> Self::Output {
        self.eval(&darray![arg.as_ref().to_owned()]).into()
    }
}

impl<T: AsRef<DA>> Eval<T> for AlgebraicVector<DA> {
    type Output = AlgebraicVector<DA>;
    fn eval(&self, arg: T) -> Self::Output {
        self.compile().eval(arg.as_ref())
    }
}

impl<T: AsRef<DA>> Eval<T> for DA {
    type Output = DA;
    fn eval(&self, arg: T) -> Self::Output {
        self.compile().eval(arg.as_ref())[0].to_owned()
    }
}

impl Eval<f64> for DA {
    type Output = f64;
    fn eval(&self, arg: f64) -> Self::Output {
        self.compile().eval(arg)[0].to_owned()
    }
}
