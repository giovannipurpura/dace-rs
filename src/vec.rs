use crate::*;
use dacecore::*;
use auto_ops::*;
pub use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::Array1;
use ndarray::*;
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{Inverse, Norm};
use num_traits::{One, Zero};
use std::convert::From;
use std::iter::zip;
use std::ops;
use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

/// Generic vector class to handle vectors
/// of algebraic types and their algebraic operations.
pub struct AlgebraicVector<T>(
    /// 1D ndarray wrapped by the AlgebraicVector
    pub Array1<T>,
);

impl<T: PartialEq> PartialEq for AlgebraicVector<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for AlgebraicVector<T> {}

impl<T> Clone for AlgebraicVector<T>
where
    OwnedRepr<T>: RawDataClone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Deref for AlgebraicVector<T> {
    type Target = Array1<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for AlgebraicVector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, U> AsRef<T> for AlgebraicVector<U>
where
    T: ?Sized,
    <AlgebraicVector<U> as Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.deref().as_ref()
    }
}

impl<T, U> AsMut<T> for AlgebraicVector<U>
where
    <AlgebraicVector<U> as Deref>::Target: AsMut<T>,
{
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut().as_mut()
    }
}

impl<T> From<Vec<T>> for AlgebraicVector<T> {
    fn from(v: Vec<T>) -> Self {
        Self(Array1::from(v))
    }
}

impl<T> From<Array1<T>> for AlgebraicVector<T> {
    fn from(v: Array1<T>) -> Self {
        Self(v)
    }
}

impl<T: Clone> AlgebraicVector<T> {
    pub fn zeros(shape: impl ShapeBuilder<Dim = Ix1>) -> Self
    where
        T: Zero,
    {
        Array1::<T>::zeros(shape).into()
    }

    pub fn ones(shape: impl ShapeBuilder<Dim = Ix1>) -> Self
    where
        T: One,
    {
        Array1::<T>::ones(shape).into()
    }
}

impl<T: Display> Display for AlgebraicVector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let last_index = self.len() - 1;
        writeln!(f, "[")?;
        for (i, el) in self.indexed_iter() {
            el.fmt(f)?;
            if i != last_index {
                writeln!(f, ",")?;
            }
        }
        writeln!(f, "]")
    }
}

impl<T: Debug> Debug for AlgebraicVector<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<T> IntoIterator for AlgebraicVector<T> {
    type Item = <Array1<T> as IntoIterator>::Item;
    type IntoIter = <Array1<T> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a AlgebraicVector<T> {
    type Item = <&'a Array1<T> as IntoIterator>::Item;
    type IntoIter = <&'a Array1<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AlgebraicVector<T> {
    type Item = <&'a mut Array1<T> as IntoIterator>::Item;
    type IntoIter = <&'a mut Array1<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, A: 'a> IntoNdProducer for &'a AlgebraicVector<A> {
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix1;
    type Output = ArrayView1<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(&self.0)
    }
}

impl AlgebraicVector<DA> {
    /// Compute the element-wise constant part of an `AlgebraicVector<DA>`.
    pub fn cons(&self) -> AlgebraicVector<f64> {
        Zip::from(self).map_collect(DA::cons).into()
    }

    /// Compute the element-wise sine of an `AlgebraicVector<DA>`.
    pub fn sin(&self) -> Self {
        Zip::from(self).map_collect(DA::sin).into()
    }

    /// Compute the element-wise cosine of an `AlgebraicVector<DA>`.
    pub fn cos(&self) -> Self {
        Zip::from(self).map_collect(DA::cos).into()
    }

    /// Compute the element-wise tangent of an `AlgebraicVector<DA>`.
    pub fn tan(&self) -> Self {
        Zip::from(self).map_collect(DA::tan).into()
    }

    /// Compute the element-wise arcsine of an `AlgebraicVector<DA>`.
    pub fn asin(&self) -> Self {
        Zip::from(self).map_collect(DA::asin).into()
    }

    /// Compute the element-wise arcsine of an `AlgebraicVector<DA>`.
    #[inline(always)]
    pub fn arcsin(&self) -> Self {
        self.sin()
    }

    /// Compute the element-wise arccosine of an `AlgebraicVector<DA>`.
    pub fn acos(&self) -> Self {
        Zip::from(self).map_collect(DA::acos).into()
    }

    /// Compute the element-wise arccosine of an `AlgebraicVector<DA>`.
    #[inline(always)]
    pub fn arccos(&self) -> Self {
        self.cos()
    }

    /// Compute the element-wise arctangent of an `AlgebraicVector<DA>`.
    pub fn atan(&self) -> Self {
        Zip::from(self).map_collect(DA::atan).into()
    }

    /// Compute the element-wise arctangent of an `AlgebraicVector<DA>`.
    #[inline(always)]
    pub fn arctan(&self) -> Self {
        self.tan()
    }

    /// Compute the element-wise four-quadrant arctangent of Y/X of two `AlgebraicVector<DA>`.
    pub fn atan2(&self, oth: &Self) -> Self {
        Zip::from(self).and(oth).map_collect(DA::atan2).into()
    }

    /// Compute the element-wise four-quadrant arctangent of Y/X of two `AlgebraicVector<DA>`.
    #[inline(always)]
    pub fn arctan2(&self, oth: &Self) -> Self {
        self.atan2(oth)
    }

    /// Get an `AlgebraicVector<DA>` with all monomials of order
    /// less than `min` and greater than `max` removed (trimmed).
    ///
    /// # Arguments
    ///
    /// * `min` - minimum order to be preserved
    /// * `max` - maximum order to be preserved
    pub fn trim(&self, imin: u32, imax: impl Into<Option<u32>>) -> Self {
        let imax = imax.into().unwrap_or_else(DA::max_order);
        Zip::from(self).map_collect(|el| el.trim(imin, imax)).into()
    }

    /// Compute the derivative of a `AlgebraicVector<T>` with respect to variable `p`.
    ///
    /// # Arguments
    ///
    /// * `p` variable with respect to which the derivative is calculated
    pub fn deriv(&self, i: u32) -> Self {
        Zip::from(self).map_collect(|el| el.deriv(i)).into()
    }

    /// Compute the integral of a `AlgebraicVector<T>` with respect to variable `p`.
    ///
    /// # Arguments
    ///
    /// * `p` variable with respect to which the integral is calculated
    pub fn integ(&self, i: u32) -> Self {
        Zip::from(self).map_collect(|el| el.integ(i)).into()
    }

    /// Partial evaluation of vector of polynomials.
    ///
    /// In each element of the vector, variable `var` is replaced by the value `val`.
    ///
    /// # Arguments
    ///
    /// * `var` - variable number to be replaced
    /// * `val` - value by which to replace the variable
    pub fn plug(&self, var: u32, val: f64) -> Self {
        Zip::from(self).map_collect(|el| el.plug(var, val)).into()
    }

    /// Get the DA identity of dimension `n`.
    ///
    /// # Arguments
    ///
    /// * `n` - dimension of the identity
    pub fn identity(n: impl Into<Option<usize>>) -> Self {
        let n = n.into().unwrap_or_else(|| DA::max_variables() as usize);
        let mut out = Self::zeros(n);
        for i in 0..n {
            unsafe { daceCreateVariable(&mut out[i], i as u32 + 1, 1.0) };
            check_exception_panic();
        }
        out
    }

    /// Return the linear part of an `AlgebraicVector<DA>`.
    pub fn linear(&self) -> AlgebraicMatrix<f64> {
        let nvar = DA::max_variables() as usize;
        let mut out = AlgebraicMatrix::zeros([self.len(), nvar]);
        for (i, el) in self.indexed_iter() {
            out.index_axis_mut(Axis(0), i).assign(&el.linear());
        }
        out
    }

    /// Compute the norm of an `AlgebraicVector<DA>`.
    ///
    /// # Panics
    ///
    /// Panics if the constant part of any element is <= 0.0.
    pub fn vnorm(&self) -> DA {
        let mut out = DA::new();
        let mut tmp = DA::new();
        unsafe {
            for el in self {
                daceSquare(el, &mut tmp);
                daceAdd(&out, &tmp, &mut out);
            }
            daceSquareRoot(&out, &mut out);
        }
        check_exception_panic();
        out
    }

    /// Normalize an `AlgebraicVector<DA>`.
    pub fn normalize(&self) -> AlgebraicVector<DA> {
        self / self.vnorm()
    }

    /// Invert the polynomials map given by the `AlgebraicVector<DA>`.
    ///
    /// # Panics
    ///
    /// Panics if the length of the vector exceeds the maximum number of DA variables.
    pub fn invert(&self) -> Result<Self, LinalgError> {
        let ord = DA::truncation_order();
        let nvar = self.len();
        let max_vars = DA::max_variables() as usize;
        if nvar > max_vars {
            panic!("Vector<DA>::inverse: length of vector exceeds maximum number of DA variables.")
        }

        // Create DA identity
        let dda = Self::identity(nvar);

        // Split map into constant part AC,
        // non-constant part M, and non-linear part AN
        let ac = self.cons();
        let m = self.trim(1, None);
        let an = m.trim(2, None);

        // Extract the linear coefficients matrix
        let al = m.linear();

        // Compute the inverse of linear coefficients matrix
        let ai = al.inv()?;

        // Compute DA representation of the inverse of the linear part
        // of the map and its composition with non-linear part AN
        let aioan = ai.dot(&an).compile();
        let linv = ai.dot(&dda);

        // Iterate to obtain the inverse map
        let mut mi = linv.clone();
        for i in 1..ord {
            DA::set_truncation_order(i + 1);
            mi = &linv - aioan.eval(&mi);
        }

        Ok(mi.eval(&(dda - ac)))
    }
}

impl Compile for AlgebraicVector<DA> {
    /// Compile current `AlgebraicVector<DA>` object and create a compiledDA object.
    fn compile(&self) -> da::CompiledDA {
        let dim = self.len();
        let mut ac = vec![0.0; (dim + 2) * unsafe { daceGetMaxMonomials() } as usize];
        let mut terms = 0;
        let mut _vars = 0;
        let mut ord = 0;
        let collection = self.iter().collect::<Vec<&DA>>();

        unsafe {
            daceEvalTree(
                collection.as_ptr(),
                dim as u32,
                ac.as_mut_ptr(),
                &mut terms,
                &mut _vars,
                &mut ord,
            );
        }
        check_exception_panic();
        da::CompiledDA {
            dim,
            ac,
            terms,
            _vars,
            ord,
        }
    }
}

/// Cross product of two objects.
pub trait Cross<'a, Rhs> {
    /// The resulting type after applying the cross operation.
    type Output;
    /// Compute the cross product.
    fn cross(&'a self, oth: &'a Rhs) -> Self::Output;
}

impl<'a, T, U> Cross<'a, AlgebraicVector<T>> for AlgebraicVector<U>
where
    T: 'a,
    U: 'a,
    &'a U: ops::Mul<&'a T>,
    <&'a U as ops::Mul<&'a T>>::Output: ops::Sub<<&'a U as ops::Mul<&'a T>>::Output>,
{
    type Output = AlgebraicVector<<<&'a U as ops::Mul<&'a T>>::Output as ops::Sub<<&'a U as ops::Mul<&'a T>>::Output>>::Output>;
    /// Computes the cross product with an `AlgebraicVector<T>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - operand (`AlgebraicVector<T>`)
    ///
    /// # Panics
    ///
    /// Panics if the lengths of the vectors are not equal to 3.
    fn cross(&'a self, other: &'a AlgebraicVector<T>) -> Self::Output {
        if self.len() != 3 || other.len() != 3 {
            panic!("DACE::AlgebraicVector<T>::cross(): Inputs must be 3 element AlgebraicVectors.");
        }
        darray![
            &self[1] * &other[2] - &self[2] * &other[1],
            &self[2] * &other[0] - &self[0] * &other[2],
            &self[0] * &other[1] - &self[1] * &other[0],
        ]
    }
}

impl AlgebraicVector<f64> {
    /// Compute the norm of an `AlgebraicVector<f64>`.
    pub fn vnorm(&self) -> f64 {
        self.norm_l2()
    }
}

impl Dot<AlgebraicVector<DA>> for AlgebraicVector<DA> {
    type Output = DA;
    /// Compute the scalar product of this `AlgebraicVector<DA>`
    /// with an `AlgebraicVector<DA>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - operand (`AlgebraicVector<DA>`)
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    fn dot(&self, rhs: &AlgebraicVector<DA>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("The elements must have the same length.");
        }
        let mut out = DA::new();
        let mut tmp: DA = DA::new();
        for (x, y) in zip(self, rhs) {
            unsafe { daceMultiply(x, y, &mut tmp) };
            check_exception_panic();
            out += &tmp;
        }
        out
    }
}

impl Dot<AlgebraicVector<f64>> for AlgebraicVector<DA> {
    type Output = DA;
    /// Compute the scalar product of this `AlgebraicVector<DA>`
    /// with an `AlgebraicVector<f64>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - operand (`AlgebraicVector<f64>`)
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    fn dot(&self, rhs: &AlgebraicVector<f64>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("The elements must have the same length.");
        }
        let mut out = DA::new();
        let mut tmp: DA = DA::new();
        for (x, y) in zip(self, rhs) {
            unsafe { daceMultiplyDouble(x, *y, &mut tmp) };
            check_exception_panic();
            out += &tmp;
        }
        out
    }
}

impl Dot<AlgebraicVector<DA>> for AlgebraicVector<f64> {
    type Output = DA;
    /// Compute the scalar product of this `AlgebraicVector<f64>`
    /// with an `AlgebraicVector<DA>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - operand (`AlgebraicVector<DA>`)
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    fn dot(&self, rhs: &AlgebraicVector<DA>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("The elements must have the same length.");
        }
        let mut out = DA::new();
        let mut tmp: DA = DA::new();
        for (x, y) in zip(self, rhs) {
            unsafe { daceMultiplyDouble(y, *x, &mut tmp) };
            check_exception_panic();
            out += &tmp;
        }
        out
    }
}

impl Dot<AlgebraicVector<f64>> for AlgebraicVector<f64> {
    type Output = f64;
    /// Compute the scalar product of this `AlgebraicVector<f64>`
    /// with an `AlgebraicVector<f64>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - operand (`AlgebraicVector<f64>`)
    ///
    /// # Panics
    ///
    /// Panics if the vectors have different lengths.
    fn dot(&self, rhs: &AlgebraicVector<f64>) -> Self::Output {
        if self.len() != rhs.len() {
            panic!("The elements must have the same length.");
        }
        self.0.dot(&rhs.0)
    }
}

impl From<&AlgebraicVector<f64>> for AlgebraicVector<DA> {
    fn from(d: &AlgebraicVector<f64>) -> Self {
        d.map(|x| da!(*x)).into()
    }
}

impl From<AlgebraicVector<f64>> for AlgebraicVector<DA> {
    fn from(d: AlgebraicVector<f64>) -> Self {
        (&d).into()
    }
}

impl From<&AlgebraicVector<u32>> for AlgebraicVector<DA> {
    fn from(d: &AlgebraicVector<u32>) -> Self {
        d.map(|x| da!(*x)).into()
    }
}

impl From<AlgebraicVector<u32>> for AlgebraicVector<DA> {
    fn from(d: AlgebraicVector<u32>) -> Self {
        (&d).into()
    }
}

impl<'a, T: Clone> From<ArrayView1<'a, T>> for AlgebraicVector<T> {
    fn from(v: ArrayView1<'a, T>) -> Self {
        AlgebraicVector(v.to_owned())
    }
}

// Add vector

impl_op_ex!(+ |a: &AlgebraicVector<DA>, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
    (&a.0 + &b.0).into()
});

impl_op_ex!(+= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<DA>| {
    a.0 += &b.0;
});

impl_op_ex_commutative!(+ |a: &AlgebraicVector<DA>, b: &AlgebraicVector<f64>| -> AlgebraicVector<DA> {
    (&a.0 + &b.0).into()
});

impl_op_ex!(+= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<f64>| {
    a.0 = &a.0 + &b.0;
});

impl_op_ex!(+ |a: &AlgebraicVector<f64>, b: &AlgebraicVector<f64>| -> AlgebraicVector<f64> {
    (&a.0 + &b.0).into()
});

impl_op_ex!(+= |a: &mut AlgebraicVector<f64>, b: &AlgebraicVector<f64>| {
    a.0 += &b.0;
});

// Add scalar

impl_op_ex_commutative!(+ |a: &AlgebraicVector<DA>, b: &DA| -> AlgebraicVector<DA> {
    (&a.0 + b.to_owned()).into()
});

impl_op_ex!(+= |a: &mut AlgebraicVector<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs += b;
    }
});

impl_op_ex_commutative!(+ |a: &AlgebraicVector<DA>, b: &f64| -> AlgebraicVector<DA> {
    (&a.0 + *b).into()
});

impl_op_ex!(+= |a: &mut AlgebraicVector<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs += b;
    }
});

impl_op_ex_commutative!(+ |a: &AlgebraicVector<f64>, b: &DA| -> AlgebraicVector<DA> {
    AlgebraicVector::<DA>::from(a) + b
});

impl_op_ex_commutative!(+ |a: &AlgebraicVector<f64>, b: &f64| -> AlgebraicVector<f64> {
    (&a.0 + *b).into()
});

impl_op_ex!(+= |a: &mut AlgebraicVector<f64>, b: &f64| {
    a.0 += *b
});

// Sub vector

impl_op_ex!(
    -|a: &AlgebraicVector<DA>, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
        (&a.0 - &b.0).into()
    }
);

impl_op_ex!(-= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<DA>| {
    a.0 -= &b.0;
});

impl_op_ex!(
    -|a: &AlgebraicVector<DA>, b: &AlgebraicVector<f64>| -> AlgebraicVector<DA> {
        (&a.0 - &b.0).into()
    }
);

impl_op_ex!(
    -|a: &AlgebraicVector<f64>, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
        (&a.0 - b.0.to_owned()).into()
    }
);

impl_op_ex!(
    -|a: &AlgebraicVector<f64>, b: &AlgebraicVector<f64>| -> AlgebraicVector<f64> {
        (&a.0 - &b.0).into()
    }
);

impl_op_ex!(-= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<f64>| {
    for (lhs, rhs) in zip(&mut a.0, &b.0) {
        *lhs -= *rhs;
    }
});

// Sub scalar

impl_op_ex!(-|a: &AlgebraicVector<DA>, b: &DA| -> AlgebraicVector<DA> {
    (&a.0 - b.to_owned()).into()
});

impl_op_ex!(-|a: &DA, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
    (-(&b.0 - a.to_owned())).into()
});

impl_op_ex!(-= |a: &mut AlgebraicVector<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs -= b;
    }
});

impl_op_ex!(-|a: &AlgebraicVector<DA>, b: &f64| -> AlgebraicVector<DA> { (&a.0 - *b).into() });

impl_op_ex!(-|a: &f64, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> { (-(&b.0 - *a)).into() });

impl_op_ex!(-= |a: &mut AlgebraicVector<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs -= b;
    }
});

impl_op_ex!(-|a: &AlgebraicVector<f64>, b: &DA| -> AlgebraicVector<DA> {
    AlgebraicVector::<DA>::from(a) - b
});

impl_op_ex!(-|a: &DA, b: &AlgebraicVector<f64>| -> AlgebraicVector<DA> {
    a - AlgebraicVector::<DA>::from(b)
});

impl_op_ex!(-|a: &AlgebraicVector<f64>, b: &f64| -> AlgebraicVector<f64> { (&a.0 - *b).into() });

impl_op_ex!(-|a: &f64, b: &AlgebraicVector<f64>| -> AlgebraicVector<f64> { (-(&b.0 - *a)).into() });

impl_op_ex!(-= |a: &mut AlgebraicVector<f64>, b: &f64| {
    a.0 -= *b;
});

// Mul vector

impl_op_ex!(
    *|a: &AlgebraicVector<DA>, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
        (&a.0 * &b.0).into()
    }
);

impl_op_ex!(*= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<DA>| {
    a.0 *= &b.0;
});

impl_op_ex_commutative!(*|a: &AlgebraicVector<DA>,
                          b: &AlgebraicVector<f64>|
 -> AlgebraicVector<DA> { (&a.0 * &b.0).into() });

impl_op_ex!(*= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<f64>| {
    for (lhs, rhs) in zip(&mut a.0, &b.0) {
        *lhs *= *rhs;
    }
});

// Mul scalar

impl_op_ex_commutative!(*|a: &AlgebraicVector<DA>, b: &DA| -> AlgebraicVector<DA> {
    (&a.0 * b.to_owned()).into()
});

impl_op_ex!(*= |a: &mut AlgebraicVector<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs *= b;
    }
});

impl_op_ex_commutative!(*|a: &AlgebraicVector<DA>, b: &f64| -> AlgebraicVector<DA> {
    (&a.0 * *b).into()
});

impl_op_ex!(*= |a: &mut AlgebraicVector<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs *= *b;
    }
});

impl_op_ex_commutative!(*|a: &AlgebraicVector<f64>, b: &DA| -> AlgebraicVector<DA> {
    AlgebraicVector::<DA>::from(a) * b
});

impl_op_ex_commutative!(
    *|a: &AlgebraicVector<f64>, b: &f64| -> AlgebraicVector<f64> { (&a.0 * *b).into() }
);

impl_op_ex!(*= |a: &mut AlgebraicVector<f64>, b: &f64| {
    a.0 *= *b;
});

// Div vector

impl_op_ex!(
    /|a: &AlgebraicVector<DA>, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
        (&a.0 / &b.0).into()
    }
);

impl_op_ex!(/= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<DA>| {
    a.0 /= &b.0;
});

impl_op_ex!(
    /|a: &AlgebraicVector<DA>, b: &AlgebraicVector<f64>| -> AlgebraicVector<DA> {
        (&a.0 / &b.0).into()
    }
);

impl_op_ex!(
    /|a: &AlgebraicVector<f64>, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
        (&a.0 / b.0.to_owned()).into()
    }
);

impl_op_ex!(/= |a: &mut AlgebraicVector<DA>, b: &AlgebraicVector<f64>| {
    for (lhs, rhs) in zip(&mut a.0, &b.0) {
        *lhs /= *rhs;
    }
});

// Div scalar

impl_op_ex!(/|a: &AlgebraicVector<DA>, b: &DA| -> AlgebraicVector<DA> {
    (&a.0 / b.to_owned()).into()
});

impl_op_ex!(/|a: &DA, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
    (Array1::<DA>::ones(b.len()) * a.to_owned() / &b.0).into()
});

impl_op_ex!(/= |a: &mut AlgebraicVector<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs /= b;
    }
});

impl_op_ex!(/|a: &AlgebraicVector<DA>, b: &f64| -> AlgebraicVector<DA> { (&a.0 / *b).into() });

impl_op_ex!(/|a: &f64, b: &AlgebraicVector<DA>| -> AlgebraicVector<DA> {
    (Array1::<DA>::ones(b.len()) * *a / &b.0).into()
});

impl_op_ex!(/= |a: &mut AlgebraicVector<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs /= *b;
    }
});

impl_op_ex!(/|a: &AlgebraicVector<f64>, b: &DA| -> AlgebraicVector<DA> {
    AlgebraicVector::<DA>::from(a) / b
});

impl_op_ex!(/|a: &DA, b: &AlgebraicVector<f64>| -> AlgebraicVector<DA> {
    a / AlgebraicVector::<DA>::from(b)
});

impl_op_ex!(/|a: &AlgebraicVector<f64>, b: &f64| -> AlgebraicVector<f64> { (&a.0 / *b).into() });

impl_op_ex!(/|a: &f64, b: &AlgebraicVector<f64>| -> AlgebraicVector<f64> {
    (*a / &b.0).into()
});

impl_op_ex!(/= |a: &mut AlgebraicVector<f64>, b: &f64| {
    a.0 /= *b;
});

// Neg

impl<T> ops::Neg for AlgebraicVector<T>
where
    AlgebraicVector<T>: ops::MulAssign<f64>,
{
    type Output = AlgebraicVector<T>;
    fn neg(mut self) -> Self::Output {
        self *= -1.0;
        self
    }
}

impl<T> ops::Neg for &AlgebraicVector<T>
where
    T: Clone + ops::Mul<f64, Output = T>,
{
    type Output = AlgebraicVector<T>;
    fn neg(self) -> Self::Output {
        (&self.0 * -1.0).into()
    }
}
