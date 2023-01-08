//! Wrapped C DACECore unsafe functions.
//!
//! These functions can be used on DA objects, considering that they are unsafe
//! and that DACE Exception checking must be manually performed.

use super::DA;
use std::os::raw::{c_char, c_double, c_int, c_uint, c_void};

extern "C" {
    /// Set up the ordering and addressing arrays in the common data structure
    /// and initialize DA memory.
    ///
    /// **MUST BE CALLED BEFORE ANY OTHER DA ROUTINE CAN BE USED.**
    ///
    /// *Also initializes the truncation order to the maximum computation order
    /// and disables the DA epsilon cutoff by setting it to 0.0.*
    ///
    /// # Arguments
    ///
    /// * `no` - order of the Taylor polynomials
    /// * `nv` - number of variables considered
    pub fn daceInitialize(no: c_uint, nv: c_uint);

    /// Set up thread local data structures at the beginning of a new thread.
    ///
    /// The main thread must call `daceInitialize` once before spawning new threads.
    /// All spawned threads must then call `daceInitializeThread` to initialize the
    /// thread before performing any other operations.
    ///
    /// `daceInitialize` MUST NOT be called again by any thread while other threads
    /// are active.
    ///
    /// *Also initializes the truncation order to the maximum computation order
    /// and disables the DA epsilon cutoff by setting it to 0.0.*
    pub fn daceInitializeThread();

    /// Clean up thread local data structures at the end of thread's life time.
    ///
    /// Each spawned thread (except for the main thread) should call `daceCleanupThread`
    /// before exitting to ensure any thread local memory is properly released.
    ///
    /// No DACE operations must be performed after calling `daceCleanupThread`.
    pub fn daceCleanupThread();

    /// Get the major, minor and patch version number of the DACE core.
    ///
    /// These values can be checked by the interface to ensure compatibility.
    ///
    /// # Arguments
    ///
    /// * `imaj` - Major version number
    /// * `imin` - Minor version number
    /// * `ipat` - Patch version number
    pub fn daceGetVersion(imaj: *mut c_int, imin: *mut c_int, ipat: *mut c_int);

    /// Set cutoff value to eps and return the previous value.
    ///
    /// # Arguments
    ///
    /// * `eps` - New cutoff value at or below which coefficients can be flushed to
    ///   zero for efficiency purposes.
    ///
    /// *__This feature can have severe unintended consequences if used incorrectly!__
    /// Flushing occurs for any intermediate result also within the DACE, and can result
    /// in wrong solutions whenever DA coefficients become very small relative to epsilon.
    /// For example, division by a large DA divisor can cause the (internally calculated)
    /// multiplicative inverse to be entirely flushed to zero, resulting in a zero DA
    /// quotient independently of the size of the dividend.*
    pub fn daceSetEpsilon(deps: c_double) -> c_double;

    /// Get the cutoff value eps.
    pub fn daceGetEpsilon() -> c_double;

    /// Get machine epsilon value.
    pub fn daceGetMachineEpsilon() -> c_double;

    /// Get the maximum computation order set in the initialization routine.
    pub fn daceGetMaxOrder() -> c_uint;

    /// Get the maximum number of variables set in the initialization routine.
    pub fn daceGetMaxVariables() -> c_uint;

    /// Get the maximum number of monomials for the current setup.
    pub fn daceGetMaxMonomials() -> c_uint;

    /// Get the current truncation order set for computations.
    pub fn daceGetTruncationOrder() -> c_uint;

    /// Set the current truncation order for future computations and
    /// return the previous truncation order.
    ///
    /// # Arguments
    ///
    /// * `fnot` - the new truncation order
    pub fn daceSetTruncationOrder(fnot: c_uint) -> c_uint;

    /// Get the current error XYY code.
    pub fn daceGetError() -> c_uint;

    /// Get the current error X code.
    pub fn daceGetErrorX() -> c_uint;

    /// Get the current error YY code.
    pub fn daceGetErrorYY() -> c_uint;

    /// Get the function name of current generated error.
    pub fn daceGetErrorFunName() -> *const c_char;

    /// Get the current error message.
    pub fn daceGetErrorMSG() -> *const c_char;

    /// Clear the error code.
    pub fn daceClearError();

    /// Allocate storage for a DA vector `inc` with memory length `len`.
    ///
    /// # Arguments
    ///
    /// * `inc` - index of the newly created variable
    /// * `len` - length of the variable to allocate.
    ///   If `len` = 0, the length is automatically determined to be large
    ///   enough for any DA vector (i.e. `len`=`nmmax`).
    pub fn daceAllocateDA(inc: *mut DA, len: c_uint);

    /// Deallocate DA vector `inc`.
    ///
    /// # Arguments
    ///
    /// * `inc` - index of the DA variable to free
    pub fn daceFreeDA(inc: *mut DA);

    /// Invalidate DA vector `inc` without deallocating associated memory.
    ///
    /// # Arguments
    ///
    /// * `inc` - index of the DA variable to invalidate
    pub fn daceInvalidateDA(inc: *mut DA);

    /// Dump information about the current memory management status to stdout.
    pub fn daceMemoryDump();

    /// Create a DA object to be `ckon` times the i-th independent variable.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to store the resulting DA in
    /// * `i` - number of the independent variable to create
    /// * `ckon` - coefficient of the independent DA variable created
    ///
    /// *Independent DA variable indices are 1-based, i.e. the first independent
    /// variable is `i`=1.
    /// The case of `i`=0 corresponds to the constant part of the polynomial.*
    pub fn daceCreateVariable(inc: *mut DA, i: c_uint, ckon: c_double);

    /// Create a DA object to be ckon times the monomial
    /// given by the exponents in `jj[]`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to store the resulting DA in
    /// * `jj` - C array with `nvmax` exponents indicating the monomial to create
    /// * `ckon` - coefficient of the monomial created
    pub fn daceCreateMonomial(ina: *mut DA, jj: *const c_uint, ckon: c_double);

    /// Create a DA object with constant part equal to `ckon`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to store the resulting DA in
    /// * `ckon` - coefficient of the constant part of the result
    pub fn daceCreateConstant(ina: *mut DA, ckon: c_double); // useless

    /// Create a DA object with all coefficients set to the constant value `ckon`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to store the resulting DA in
    /// * `ckon` - coefficient of the monomials
    pub fn daceCreateFilled(ina: *mut DA, ckon: c_double);

    /// Create a DA object with randomly filled coefficients.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to store the resulting DA in
    /// * `cm` - The filling factor between -1.0 and 1.0.
    ///
    /// The absolute value of the filling factor determines the fraction of non-zero
    /// coefficients.
    /// If `cm` is positive, the values are weighted by order such that
    /// the coefficients decay exponentially with the order from 1.0 towards the
    /// machine epsilon in the highest order.
    /// If `cm` is negative, all coefficients are chosen to be between -1.0 and 1.0.
    pub fn daceCreateRandom(ina: *mut DA, cm: c_double);

    /// Extract the constant part from a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to extract constant part from
    pub fn daceGetConstant(ina: *const DA) -> c_double;

    /// Extract the linear part of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to extract linear part from
    /// * `c` - C array of length `nvmax` containing the linear coefficients in order
    pub fn daceGetLinear(ina: *const DA, c: *mut c_double);

    /// Extract coefficient of a monomial in a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to extract monomial coefficient from
    /// * `jj` - C array of `nvmax` exponents identifying the monomial
    pub fn daceGetCoefficient(ina: *const DA, jj: *const c_uint) -> c_double;

    /// Extract coefficient of a monomial in a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to extract monomial coefficient from
    /// * `ic` - DA coding integer of the monomial to extract
    pub fn daceGetCoefficient0(ina: *const DA, ic: c_uint) -> c_double;

    /// Set coefficient of a monomial in a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to set monomial in
    /// * `jj` - C array of `nvmax` exponents identifying the monomial
    /// * `cjj` - Value of the corresponding coefficient
    pub fn daceSetCoefficient(ina: *mut DA, jj: *const c_uint, cjj: c_double);

    /// Set coefficient of a monomial in a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to set monomial in
    /// * `ic` - DA coding integer of the monomial to set
    /// * `cjj` - Value of the corresponding coefficient
    pub fn daceSetCoefficient0(ina: *mut DA, ic: c_uint, cjj: c_double);

    /// Extract coefficient at position npos (starting with 1) in the list of
    /// non-zero coefficients in the DA object and return its exponents and
    /// coefficient. If the monomial does not exist, the value 0.0 is returned.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to extract monomial from
    /// * `npos` - Index of the monomial to extract
    /// * `jj` - C array of `nvmax` elements for returning the exponents of the monomial
    /// * `cjj` - Pointer where to store the value of the coefficient of the monomial
    pub fn daceGetCoefficientAt(ina: *const DA, npos: c_uint, jj: *mut c_uint, cjj: *mut c_double);

    /// Return the number of non-zero monomials in a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to get length of
    pub fn daceGetLength(ina: *const DA) -> c_uint;

    /// Copy content of one DA object into another DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to copy from
    /// * `inb` - Pointer to DA object to copy to
    pub fn daceCopy(ina: *const DA, inb: *mut DA);

    /// Copy content of one DA object into another DA object filtering out terms
    /// below a certain threshold.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to copy from
    /// * `inb` - Pointer to DA object to copy to
    ///
    /// *This routine is slightly worse than non-filtering version (about 10%)*
    pub fn daceCopyFiltering(ina: *const DA, inb: *mut DA);

    /// Copy monomials from a DA object `ina` to DA object `inb` if the same monomial
    /// is non-zero in DA object `inc`, while filtering out terms below the current
    /// cutoff
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to filter
    /// * `inb` - Pointer to DA object to store the filtered result in
    /// * `inc` - Pointer to DA object providing the filter template
    pub fn daceFilter(ina: *const DA, inb: *mut DA, inc: *const DA);

    /// Truncate a DA object to contain only terms of order larger or equal to `imin`
    /// and less than or equal `imax`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to trim
    /// * `imin` - Minimum order to keep
    /// * `imax` - Maximum order to keep
    /// * `inc` - Pointer to DA object to store the truncated result in
    pub fn daceTrim(ina: *const DA, imin: c_uint, imax: c_uint, inc: *mut DA);

    /// Compute the weighted sum of two DA objects.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `afac` - Weighting factor to multiply `ina` by
    /// * `inb` - Pointer to the second DA object to operate on
    /// * `bfac` - Weighting factor to multiply `inb` by
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is __NOT__ aliasing safe! So `inc` MUST BE DIFFERENT from `ina` and `inb`.*
    pub fn daceWeightedSum(ina: *const DA, afac: c_double, inb: *const DA, bfac: c_double, inc: *mut DA);

    /// Perform addition of two DA objects.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `inb` - Pointer to the first DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina` or `inb`.*
    pub fn daceAdd(ina: *const DA, inb: *const DA, inc: *mut DA);

    /// Perform subtraction of two DA objects.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `inb` - Pointer to the first DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina` or `inb`.*
    pub fn daceSubtract(ina: *const DA, inb: *const DA, inc: *mut DA);

    /// Perform multiplication of two DA objects.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `inb` - Pointer to the first DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina` or `inb`.*
    pub fn daceMultiply(ina: *const DA, inb: *const DA, inc: *mut DA);

    /// Multiply two DA vectors component-wise,
    /// i.e. each monomial of `ina` with the corresponding monomial of `inb`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `inb` - Pointer to the first DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina` or `inb`.*
    pub fn daceMultiplyMonomials(ina: *const DA, inb: *const DA, inc: *mut DA);

    /// Perform division of two DA objects.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `inb` - Pointer to the first DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina` or `inb`.*
    pub fn daceDivide(ina: *const DA, inb: *const DA, inc: *mut DA);

    /// Square a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to square
    /// * `inb` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceSquare(ina: *const DA, inb: *mut DA);

    /// Add constant to a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `ckon` - Constant value to add
    /// * `inb` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inb` can be the same as `ina`.*
    pub fn daceAddDouble(ina: *const DA, ckon: c_double, inb: *mut DA);

    /// Subtract DA object from constant.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `ckon` - Constant value to subtract from
    /// * `inb` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inb` can be the same as `ina`.*
    pub fn daceDoubleSubtract(ina: *const DA, ckon: c_double, inb: *mut DA);

    /// Subtract constant from a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `ckon` - Constant value to subtract
    /// * `inb` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inb` can be the same as `ina`.*
    pub fn daceSubtractDouble(ina: *const DA, ckon: c_double, inb: *mut DA);

    /// Multiply constant and DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `ckon` - Constant value to multiply by
    /// * `inb` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inb` can be the same as `ina`.*
    pub fn daceMultiplyDouble(ina: *const DA, ckon: c_double, inb: *mut DA);

    /// Divide DA object by a constant.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `ckon` - Constant value to divide by
    /// * `inb` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inb` can be the same as `ina`.*
    pub fn daceDivideDouble(ina: *const DA, ckon: c_double, inb: *mut DA);

    /// Divide constant by DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `ckon` - Constant value to divide
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceDoubleDivide(ina: *const DA, ckon: c_double, inb: *mut DA);

    /// Divide a DA vector by a single variable to some power, if possible.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `var` - Number of the independent variable by which to divide
    /// * `p` - Power of independent variable
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceDivideByVariable(ina: *const DA, var: c_uint, p: c_uint, inc: *mut DA);

    /// Derivative of DA object with respect to a given independent variable.
    ///
    /// # Arguments
    ///
    /// * `idif` - Number of the independent variable with respect to which the derivative is taken
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceDifferentiate(idif: c_uint, ina: *const DA, inc: *mut DA);

    /// Integral of DA object with respect to a given independent variable.
    ///
    /// # Arguments
    ///
    /// * `iint` - Number of the independent variable with respect to which the integral is taken
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceIntegrate(iint: c_uint, ina: *const DA, inc: *mut DA);

    /// Truncate the constant part of a DA object to an integer.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceTruncate(ina: *const DA, inc: *mut DA);

    /// Round the constant part of a DA object to an integer.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceRound(ina: *const DA, inc: *mut DA);

    /// Modulo the constant part of a DA object by `p`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `p` - Value with respect to which to compute the modulo
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceModulo(ina: *const DA, p: c_double, inc: *mut DA);

    /// Raise a DA object to the p-th power.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `p` - Power to which to raise the DA object
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn dacePowerDouble(ina: *const DA, p: c_double, inc: *mut DA);

    /// Raise a DA object to the `p`-th integer power.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `np` - Power to which to raise the DA object
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn dacePower(ina: *const DA, np: c_int, inc: *mut DA);

    /// Take the `np`-th root of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `np` - Root to take of the DA object
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceRoot(ina: *const DA, np: c_int, inc: *mut DA);

    /// Compute the multiplicative inverse of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceMultiplicativeInverse(ina: *const DA, inc: *mut DA);

    /// Compute the square root of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceSquareRoot(ina: *const DA, inc: *mut DA);

    /// Compute the inverse square root of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceInverseSquareRoot(ina: *const DA, inc: *mut DA);

    /// Compute the cubic root of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceCubicRoot(ina: *const DA, inc: *mut DA);

    /// Compute the inverse cubic root of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceInverseCubicRoot(ina: *const DA, inc: *mut DA);

    /// Compute the hypothenuse of two DA objects.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `inb` - Pointer to the second DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina` or `inb`.*
    pub fn daceHypotenuse(ina: *const DA, inb: *const DA, inc: *mut DA);

    /// Compute the exponential of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceExponential(ina: *const DA, inc: *mut DA);

    /// Compute the natural logarithm root of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceLogarithm(ina: *const DA, inc: *mut DA);

    /// Compute the logarithm with respect to base `b` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `b` - Base of the logarithm to use
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceLogarithmBase(ina: *const DA, b: c_double, inc: *mut DA);

    /// Compute the decadic logarithm of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceLogarithm10(ina: *const DA, inc: *mut DA);

    /// Compute the binary logarithm of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceLogarithm2(ina: *const DA, inc: *mut DA);

    /// Compute the sine of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceSine(ina: *const DA, inc: *mut DA);

    /// Compute the cosine of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceCosine(ina: *const DA, inc: *mut DA);

    /// Compute the tangent of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceTangent(ina: *const DA, inc: *mut DA);

    /// Compute the arcsine of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceArcSine(ina: *const DA, inc: *mut DA);

    /// Compute the arccosine of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceArcCosine(ina: *const DA, inc: *mut DA);

    /// Compute the arctangent of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceArcTangent(ina: *const DA, inc: *mut DA);

    /// Arctangent of `ina`/`inb` with proper sign in \[-pi, pi\].
    ///
    /// This function follows the C standard atan2(y, x) function syntax.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the first DA object to operate on
    /// * `inb` - Pointer to the second DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceArcTangent2(ina: *const DA, inb: *const DA, inc: *mut DA);

    /// Compute the hyperbolic sine of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceHyperbolicSine(ina: *const DA, inc: *mut DA);

    /// Compute the hyperbolic cosine of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceHyperbolicCosine(ina: *const DA, inc: *mut DA);

    /// Compute the hyperbolic tangent of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceHyperbolicTangent(ina: *const DA, inc: *mut DA);

    /// Compute the hyperbolic arcsince of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceHyperbolicArcSine(ina: *const DA, inc: *mut DA);

    /// Compute the hyperbolic arccosine of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceHyperbolicArcCosine(ina: *const DA, inc: *mut DA);

    /// Compute the hyperbolic arctangent of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceHyperbolicArcTangent(ina: *const DA, inc: *mut DA);

    /// Compute the error function of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceErrorFunction(ina: *const DA, inc: *mut DA);

    /// Compute the complementary error function of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceComplementaryErrorFunction(ina: *const DA, inc: *mut DA);

    /// Compute the modified Bessel function `I_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on (constant part >= 0)
    /// * `n` - Order of the Bessel function
    /// * `scaled` - If true, the scaled Bessel function is computed (i.e. exp(-x)*I_n(x))
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceBesselIFunction(ina: *const DA, n: c_int, scaled: bool, inc: *mut DA);

    /// Compute the Bessel function `J_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on (constant part >= 0)
    /// * `n` - Order of the Bessel function
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceBesselJFunction(ina: *const DA, n: c_int, inc: *mut DA);

    /// Compute the modified Bessel function `K_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on (constant part >= 0)
    /// * `n` - Order of the Bessel function
    /// * `scaled` - If true, the scaled Bessel function is computed (i.e. exp(x)*K_n(x))
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceBesselKFunction(ina: *const DA, n: c_int, scaled: bool, inc: *mut DA);

    /// Compute the Bessel function `Y_n` of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on (constant part >= 0)
    /// * `n` - Order of the Bessel function
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceBesselYFunction(ina: *const DA, n: c_int, inc: *mut DA);

    /// Compute the Logarithmic Gamma function of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on (constant part != 0, -1, -2, ...)
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceLogGammaFunction(ina: *const DA, inc: *mut DA);

    /// Compute the Gamma function of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on (constant part != 0, -1, -2, ...)
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn daceGammaFunction(ina: *const DA, inc: *mut DA);

    /// Compute the `n`-th Psi function (i.e. the `n`+1 derivative of the logarithmic gamma function) of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to operate on (constant part != 0, -1, -2, ...)
    /// * `n` - Order of the Psi function (n >= 0)
    /// * `inc` - Pointer to the DA object to store the result in
    ///
    /// *This routine is aliasing safe, i.e. `inc` can be the same as `ina`.*
    pub fn dacePsiFunction(ina: *const DA, n: c_uint, inc: *mut DA);

    /// Compute the absolute value (maximum coefficient norm) of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to take absolute value of
    pub fn daceAbsoluteValue(ina: *const DA) -> c_double;

    /// Compute a norm of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to take norm of
    /// * `ityp` - Type of norm to compute.
    ///   - 0 = max norm
    ///   - 1 = sum norm
    ///   - oth = corresponding vector norm
    pub fn daceNorm(ina: *const DA, ityp: c_uint) -> c_double;

    /// Compute an order sorted norm of a DA object.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to take norm of
    /// * `ivar` - Independent variable with respect to which to group.
    ///   - 0 = group by monomial order
    ///   - oth = group by given independent variable
    /// * `ityp` - Type of norm to compute.
    ///   - 0 = max norm
    ///   - 1 = sum norm
    ///   - oth = corresponding vector norm
    /// * `onorm` - C array of length `nomax`+1 containing the grouped estimates
    pub fn daceOrderedNorm(ina: *const DA, ivar: c_uint, ityp: c_uint, onorm: *mut c_double);

    /// Estimate order sorted norms of DA object `ina` up to given order.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to take norm of
    /// * `ivar` - Independent variable with respect to which to group.
    ///   - 0 = group by monomial order
    ///   - oth = group by given independent variable
    /// * `ityp` - Type of norm to compute.
    ///   - 0 = max norm
    ///   - 1 = sum norm
    ///   - oth = corresponding vector norm
    /// * `nc` - Maximum order to estimate
    /// * `c` - C array of length nc+1 containing the grouped estimates
    /// * `err` - C array of length `min(nc, nomax) + 1` containing the residuals
    ///    of the exponential fit at each order. If NULL is passed in, no residuals
    ///    are computed and returned.
    ///
    /// *If estimation is not possible, zero is returned for all
    /// requested orders. in most cases this is actually not too far off.*
    pub fn daceEstimate(ina: *const DA, ivar: c_uint, ityp: c_uint, c: *mut c_double, e: *mut c_double, nc: c_uint);

    /// Compute an upper and lower bound of DA object `ina` over the domain \[-1,1\]^n.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to bound
    /// * `alo` - Pointer where to store the lower bound
    /// * `aup` - Pointer where to store the upper bound
    pub fn daceGetBounds(ina: *const DA, alo: &mut c_double, aup: &mut c_double);

    /// Evaluate DA object `ina` by providing the value to use for each monomial in DA object `inb`.
    ///
    /// This is equivalent to a monomial-wise DA dot product.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to first DA object to evaluate
    /// * `inb` - Pointer to second DA object to provide monomial values
    pub fn daceEvalMonomials(ina: *const DA, inb: *const DA) -> c_double;

    /// Replace independent variable with index `from` by `val` times the independent
    /// variable with index `to`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to evaluate
    /// * `from` - Number of the independent variable to replace
    /// * `to` - Number of the independent variable to change to
    /// * `val` - Value to scale new independent variable with
    /// * `inc` - Pointer to DA object to store the result of the replacement
    pub fn daceReplaceVariable(ina: *const DA, from: c_uint, to: c_uint, val: c_double, inc: *mut DA);

    /// Perform partial evaluation of DA object `ina` by replacing independent variable
    /// number `nvar` by the value `val`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to evaluate
    /// * `nvar` - Number of the independent variable to replace (one-based)
    /// * `val` - Value to replace independent variable with
    /// * `inc` - Pointer to DA object to store the result of the partial evaluation
    pub fn daceEvalVariable(ina: *const DA, nvar: c_uint, val: c_double, inc: *mut DA);

    /// Scale independent variable `nvar` by `val`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to scale
    /// * `nvar` - Number of the independent variable to scale
    /// * `val` - Value to scale independent variable with
    /// * `inc` - Pointer to DA object to store the result of the scaling
    pub fn daceScaleVariable(ina: *const DA, nvar: c_uint, val: c_double, inc: *mut DA);

    /// Translate independent variable `nvar` to `(a*x + c)`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to DA object to translate
    /// * `nvar` - Number of the independent variable to translate
    /// * `a` - Linear value to scale independent variable by
    /// * `c` - Constant value to translate independent variable by
    /// * `inc` - Pointer to DA object to store the result of the translation
    pub fn daceTranslateVariable(ina: *const DA, nvar: c_uint, a: c_double, c: c_double, inc: *mut DA);

    /// Compute an evaluation tree to efficiently evaluate several DA objects.
    ///
    /// # Arguments
    ///
    /// * `das` - C array of pointers to DA objects to evaluate
    /// * `count` - Number of DA objects in `das[]`
    /// * `ac` - C array of doubles containing compiled coefficients
    /// * `nterm` - Pointer where to store the total number of terms in evaluation tree
    /// * `nvar` - Pointer where to store the total number of variables in evaluation tree
    /// * `nord` - Pointer where to store the maximum order in evaluation tree
    pub fn daceEvalTree(ina: *const &DA, count: c_uint, ac: *mut c_double, nterm: *mut c_uint, nvar: *mut c_uint, nord: *mut c_uint);

    /// Print the DA object `ina` to string strs (of line length `DACE_STRLEN`).
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to be printed
    /// * `strs` - C array of size (nmmax+2)*`DACE_STRLEN` containing the
    ///   zero-terminated lines of length `DACE_STRLEN`
    /// * `nstrs` - Pointer where to store the final number of strings printed
    ///
    /// *The format of the output is written in DACE format which is loosely
    /// based on the format used by COSY INFINITY but is not fully compatible to it.*
    pub fn daceWrite(ina: *const DA, strs: *mut c_char, nstrs: *mut c_uint);

    /// Read a DA object `ina` from a human readable string representation in `strs`,
    /// containing `nstrs` contiguous zero-terminated lines of line length `DACE_STRLEN`.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to read into
    /// * `strs` - C array of size nstrs*`DACE_STRLEN` containing the
    ///    zero-terminated lines of length `DACE_STRLEN`
    /// * `nstrs` - Number of lines in strs
    ///
    /// *This routine can read both DACE output strings as well as some COSY INFINITY
    /// formated strings. COSY INFINITY input is limited to less than 16 variables
    /// (i.e. a single line per coefficient).*
    pub fn daceRead(ina: *mut DA, strs: *const c_char, nstrs: c_uint);

    /// Print a DA object `ina` to the standard output.
    ///
    /// # Arguments
    ///
    /// * `ina` - Pointer to the DA object to printed
    pub fn dacePrint(inc: *const DA);

    /// Export a DA object in a binary format.
    ///
    /// Returns 0 if the entire DA object was exported successfully or the
    /// number of truncated monomials if the object was truncated.
    ///
    /// *If blob is NULL, the value returned in size is the size (in bytes) of
    /// the memory needed to store the entire DA object.*
    ///
    /// The binary data is not supposed to be modified and its format is considered
    /// internal to the DACE. It is guaranteed that a binary representation can
    /// be read back into a DA object even with differently initialized settings.
    ///
    /// # Arguments
    ///
    /// * `ina` - The DA object to export
    /// * `blob` - Pointer to memory where to store the data
    /// * `size` - On input contains the size (in bytes) of the memory
    ///   pointed to by blob. On output contains the actual amount of memory used.
    pub fn daceExportBlob(ina: *const DA, blob: *mut c_void, size: *mut c_uint) -> c_uint;

    /// Determine the total size (in bytes) of the DACE blob.
    ///
    /// On error, 0 is returned (e.g. when the data pointed to by blob is not a DACE blob at all).
    ///
    /// If `blob` is `NULL`, the minimum size (in bytes) of a blob
    /// (i.e. the blob header size) is returned.
    ///
    /// # Arguments
    ///
    /// * `blob` - Pointer to memory where the data is stored
    ///
    /// *If called with `blob`==`NULL`, the routine will return the minimum size
    /// of data that must be read in order to determine the total size. A user can
    /// therefore call this routine twice: first with `NULL` to determine the size
    /// of the blob header to read, and then a second time with the header to determine
    /// the size of the remaining data.*
    pub fn daceBlobSize(blob: *const c_void) -> c_uint;

    /// Import a DA object in a binary format.
    ///
    /// # Arguments
    ///
    /// * `blob` - Pointer to memory where the data is stored
    /// * `inc` - The DA object to import into
    ///
    /// *This routine will silently truncate orders above the currently set
    /// maximum computation order as well as any extra variables present.*
    pub fn daceImportBlob(blob: *const c_void, inc: *mut DA);

    /// Get a pseudo-random number between 0.0 and 1.0.
    pub fn daceRandom() -> c_double;
}
