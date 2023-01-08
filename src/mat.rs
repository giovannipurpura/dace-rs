use crate::*;
use dacecore::*;
use impl_ops::*;
pub use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray::*;
use ndarray_linalg::Inverse;
use num_traits::{One, Zero};
use std::convert::From;
use std::iter::zip;
use std::ops;
use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

/// Class to handle matrices and their operations.
pub struct AlgebraicMatrix<T>(
    /// 2D ndarray wrapped by the AlgebraicMatrix
    pub Array2<T>,
);

impl<T: PartialEq> PartialEq for AlgebraicMatrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq> Eq for AlgebraicMatrix<T> {}

impl<T> Clone for AlgebraicMatrix<T>
where
    OwnedRepr<T>: RawDataClone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Deref for AlgebraicMatrix<T> {
    type Target = Array2<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for AlgebraicMatrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, U> AsRef<T> for AlgebraicMatrix<U>
where
    T: ?Sized,
    <AlgebraicMatrix<U> as Deref>::Target: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        self.deref().as_ref()
    }
}

impl<T, U> AsMut<T> for AlgebraicMatrix<U>
where
    <AlgebraicMatrix<U> as Deref>::Target: AsMut<T>,
{
    fn as_mut(&mut self) -> &mut T {
        self.deref_mut().as_mut()
    }
}

impl<A, V> From<Vec<V>> for AlgebraicMatrix<A>
where
    V: ndarray::FixedInitializer<Elem = A>,
{
    fn from(v: Vec<V>) -> Self {
        Self(Array2::from(v))
    }
}

impl<T> From<Array2<T>> for AlgebraicMatrix<T> {
    fn from(v: Array2<T>) -> Self {
        Self(v)
    }
}

impl<T: Clone> AlgebraicMatrix<T> {
    pub fn zeros(shape: impl ShapeBuilder<Dim = Ix2>) -> Self
    where
        T: Zero,
    {
        Array2::<T>::zeros(shape).into()
    }

    pub fn ones(shape: impl ShapeBuilder<Dim = Ix2>) -> Self
    where
        T: One,
    {
        Array2::<T>::ones(shape).into()
    }
}

impl<T: Display> Display for AlgebraicMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let last_index_i = self.nrows() - 1;
        let last_index_j = self.ncols() - 1;
        writeln!(f, "[")?;
        for ((i, j), el) in self.indexed_iter() {
            if j == 0 {
                writeln!(f, "  [")?;
            }
            el.fmt(f)?;
            if j == last_index_j {
                writeln!(f, "  ]")?;
                if i != last_index_i {
                    writeln!(f, ",")?;
                }
            } else {
                writeln!(f, "  ,")?;
            }
        }
        writeln!(f, "]")
    }
}

impl<T: Debug> Debug for AlgebraicMatrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<T> IntoIterator for AlgebraicMatrix<T> {
    type Item = <Array2<T> as IntoIterator>::Item;
    type IntoIter = <Array2<T> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a AlgebraicMatrix<T> {
    type Item = <&'a Array2<T> as IntoIterator>::Item;
    type IntoIter = <&'a Array2<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AlgebraicMatrix<T> {
    type Item = <&'a mut Array2<T> as IntoIterator>::Item;
    type IntoIter = <&'a mut Array2<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, A: 'a> IntoNdProducer for &'a AlgebraicMatrix<A> {
    type Item = <Self::Output as NdProducer>::Item;
    type Dim = Ix2;
    type Output = ArrayView2<'a, A>;
    fn into_producer(self) -> Self::Output {
        <_>::from(&self.0)
    }
}

impl AlgebraicMatrix<DA> {
    /// Compute the element-wise constant part of an `AlgebraicMatrix<DA>`.
    pub fn cons(&self) -> AlgebraicMatrix<f64> {
        Zip::from(self).map_collect(DA::cons).into()
    }

    /// Compute the element-wise sine of an `AlgebraicMatrix<DA>`.
    pub fn sin(&self) -> Self {
        Zip::from(self).map_collect(DA::sin).into()
    }

    /// Compute the element-wise cosine of an `AlgebraicMatrix<DA>`.
    pub fn cos(&self) -> Self {
        Zip::from(self).map_collect(DA::cos).into()
    }

    /// Compute the element-wise tangent of an `AlgebraicMatrix<DA>`.
    pub fn tan(&self) -> Self {
        Zip::from(self).map_collect(DA::tan).into()
    }

    /// Compute the element-wise arcsine of an `AlgebraicMatrix<DA>`.
    pub fn asin(&self) -> Self {
        Zip::from(self).map_collect(DA::asin).into()
    }

    /// Compute the element-wise arcsine of an `AlgebraicMatrix<DA>`.
    #[inline(always)]
    pub fn arcsin(&self) -> Self {
        self.sin()
    }

    /// Compute the element-wise arccosine of an `AlgebraicMatrix<DA>`.
    pub fn acos(&self) -> Self {
        Zip::from(self).map_collect(DA::acos).into()
    }

    /// Compute the element-wise arccosine of an `AlgebraicMatrix<DA>`.
    #[inline(always)]
    pub fn arccos(&self) -> Self {
        self.cos()
    }

    /// Compute the element-wise arctangent of an `AlgebraicMatrix<DA>`.
    pub fn atan(&self) -> Self {
        Zip::from(self).map_collect(DA::atan).into()
    }

    /// Compute the element-wise arctangent of an `AlgebraicMatrix<DA>`.
    #[inline(always)]
    pub fn arctan(&self) -> Self {
        self.tan()
    }

    /// Compute the element-wise four-quadrant arctangent of Y/X of two `AlgebraicMatrix<DA>`.
    pub fn atan2(&self, oth: &Self) -> Self {
        Zip::from(self).and(oth).map_collect(DA::atan2).into()
    }

    /// Compute the element-wise four-quadrant arctangent of Y/X of two `AlgebraicMatrix<DA>`.
    #[inline(always)]
    pub fn arctan2(&self, oth: &Self) -> Self {
        self.atan2(oth)
    }

    /// Get an `AlgebraicMatrix<DA>` with all monomials of order
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

    /// Compute the derivative of a `AlgebraicMatrix<T>` with respect to variable `p`.
    ///
    /// # Arguments
    ///
    /// * `p` variable with respect to which the derivative is calculated
    pub fn deriv(&self, i: u32) -> Self {
        Zip::from(self).map_collect(|el| el.deriv(i)).into()
    }

    /// Compute the integral of a `AlgebraicMatrix<T>` with respect to variable `p`.
    ///
    /// # Arguments
    ///
    /// * `p` variable with respect to which the integral is calculated
    pub fn integ(&self, i: u32) -> Self {
        Zip::from(self).map_collect(|el| el.integ(i)).into()
    }

    /// Partial evaluation of matrix of polynomials.
    ///
    /// In each element of the matrix, variable `var` is replaced by the value `val`.
    ///
    /// # Arguments
    ///
    /// * `var` - variable number to be replaced
    /// * `val` - value by which to replace the variable
    pub fn plug(&self, var: u32, val: f64) -> Self {
        Zip::from(self).map_collect(|el| el.plug(var, val)).into()
    }
}

impl Dot<AlgebraicVector<DA>> for AlgebraicMatrix<DA> {
    type Output = AlgebraicVector<DA>;
    /// Compute the matrix multiplication with `AlgebraicVector<DA>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicVector<DA>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicVector<DA>) -> Self::Output {
        if self.ncols() != rhs.len() {
            panic!("The elements must have compatible dimensions.")
        }
        let mut out = AlgebraicVector::<DA>::zeros(self.ncols());
        let mut tmp = DA::new();
        for i in 0..self.ncols() {
            for j in 0..self.nrows() {
                unsafe { daceMultiply(&self[[j, i]], &rhs[j], &mut tmp) };
                check_exception_panic();
                out[i] += &tmp;
            }
        }
        out
    }
}

impl Dot<AlgebraicVector<DA>> for AlgebraicMatrix<f64> {
    type Output = AlgebraicVector<DA>;
    /// Compute the matrix multiplication with `AlgebraicVector<DA>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicVector<DA>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicVector<DA>) -> Self::Output {
        if self.ncols() != rhs.len() {
            panic!("The elements must have compatible dimensions.")
        }
        let mut out = AlgebraicVector::<DA>::zeros(self.ncols());
        let mut tmp = DA::new();
        for i in 0..self.ncols() {
            for j in 0..self.nrows() {
                unsafe { daceMultiplyDouble(&rhs[j], self[[j, i]], &mut tmp) };
                check_exception_panic();
                out[i] += &tmp;
            }
        }
        out
    }
}

impl Dot<AlgebraicVector<f64>> for AlgebraicMatrix<DA> {
    type Output = AlgebraicVector<DA>;
    /// Compute the matrix multiplication with `AlgebraicVector<f64>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicVector<f64>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicVector<f64>) -> Self::Output {
        if self.ncols() != rhs.len() {
            panic!("The elements must have compatible dimensions.")
        }
        let mut out = AlgebraicVector::<DA>::zeros(self.ncols());
        let mut tmp = DA::new();
        for i in 0..self.ncols() {
            for j in 0..self.nrows() {
                unsafe { daceMultiplyDouble(&self[[j, i]], rhs[j], &mut tmp) };
                check_exception_panic();
                out[i] += &tmp;
            }
        }
        out
    }
}

impl Dot<AlgebraicVector<f64>> for AlgebraicMatrix<f64> {
    type Output = AlgebraicVector<f64>;
    /// Compute the matrix multiplication with `AlgebraicVector<f64>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicVector<f64>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicVector<f64>) -> Self::Output {
        if self.ncols() != rhs.len() {
            panic!("The elements must have compatible dimensions.")
        }
        self.0.dot(&rhs.0).into()
    }
}

impl<T, U> Dot<AlgebraicMatrix<T>> for AlgebraicVector<U>
where
    AlgebraicMatrix<T>: Dot<AlgebraicVector<U>>,
    T: Clone,
{
    type Output = <AlgebraicMatrix<T> as Dot<AlgebraicVector<U>>>::Output;
    /// Compute the matrix multiplication with `AlgebraicMatrix<T>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicMatrix<T>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicMatrix<T>) -> Self::Output {
        if rhs.nrows() != self.len() {
            panic!("The elements must have compatible dimensions.")
        }
        let mut m = rhs.clone();
        m.swap_axes(0, 1);
        m.dot(self)
    }
}

impl Dot<AlgebraicMatrix<DA>> for AlgebraicMatrix<DA> {
    type Output = AlgebraicMatrix<DA>;
    /// Compute the matrix multiplication with `AlgebraicMatrix<DA>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicMatrix<DA>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicMatrix<DA>) -> Self::Output {
        // check that first matrix columns and second matrix row are equal
        if self.ncols() != rhs.nrows() {
            panic!("DACE::AlgebraicMatrix<T>::operator*: Number of columns of first matrix must be equal to number of rows of second matrix.");
        }
        let n = self.nrows();
        let m = self.ncols();
        let p = rhs.ncols();

        let mut out = AlgebraicMatrix::zeros([n, p]);
        let mut tmp = DA::new();
        for i in 0..n {
            for j in 0..p {
                for k in 0..m {
                    unsafe { daceMultiply(&self[[i, k]], &rhs[[k, j]], &mut tmp) };
                    check_exception_panic();
                    out[[i, j]] += &tmp;
                }
            }
        }
        out
    }
}

impl Dot<AlgebraicMatrix<DA>> for AlgebraicMatrix<f64> {
    type Output = AlgebraicMatrix<DA>;
    /// Compute the matrix multiplication with `AlgebraicMatrix<DA>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicMatrix<DA>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicMatrix<DA>) -> Self::Output {
        // check that first matrix columns and second matrix row are equal
        if self.ncols() != rhs.nrows() {
            panic!("DACE::AlgebraicMatrix<T>::operator*: Number of columns of first matrix must be equal to number of rows of second matrix.");
        }
        let n = self.nrows();
        let m = self.ncols();
        let p = rhs.ncols();

        let mut out = AlgebraicMatrix::zeros([n, p]);
        let mut tmp = DA::new();
        for i in 0..n {
            for j in 0..p {
                for k in 0..m {
                    unsafe { daceMultiplyDouble(&rhs[[k, j]], self[[i, k]], &mut tmp) };
                    check_exception_panic();
                    out[[i, j]] += &tmp;
                }
            }
        }
        out
    }
}

impl Dot<AlgebraicMatrix<f64>> for AlgebraicMatrix<DA> {
    type Output = AlgebraicMatrix<DA>;
    /// Compute the matrix multiplication with `AlgebraicMatrix<f64>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicMatrix<f64>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicMatrix<f64>) -> Self::Output {
        // check that first matrix columns and second matrix row are equal
        if self.ncols() != rhs.nrows() {
            panic!("DACE::AlgebraicMatrix<T>::operator*: Number of columns of first matrix must be equal to number of rows of second matrix.");
        }
        let n = self.nrows();
        let m = self.ncols();
        let p = rhs.ncols();

        let mut out = AlgebraicMatrix::zeros([n, p]);
        let mut tmp = DA::new();
        for i in 0..n {
            for j in 0..p {
                for k in 0..m {
                    unsafe { daceMultiplyDouble(&self[[i, k]], rhs[[k, j]], &mut tmp) };
                    check_exception_panic();
                    out[[i, j]] += &tmp;
                }
            }
        }
        out
    }
}

impl Dot<AlgebraicMatrix<f64>> for AlgebraicMatrix<f64> {
    type Output = AlgebraicMatrix<f64>;
    /// Compute the matrix multiplication with `AlgebraicMatrix<f64>`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - `AlgebraicMatrix<f64>` to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the dimensions are not compatible.
    fn dot(&self, rhs: &AlgebraicMatrix<f64>) -> Self::Output {
        // check that first matrix columns and second matrix row are equal
        if self.ncols() != rhs.nrows() {
            panic!("DACE::AlgebraicMatrix<T>::operator*: Number of columns of first matrix must be equal to number of rows of second matrix.");
        }
        self.0.dot(&rhs.0).into()
    }
}

impl Inverse for AlgebraicMatrix<f64> {
    type Output = Self;
    fn inv(&self) -> ndarray_linalg::error::Result<Self::Output> {
        Ok(AlgebraicMatrix(self.0.inv()?))
    }
}

impl From<&AlgebraicMatrix<f64>> for AlgebraicMatrix<DA> {
    fn from(d: &AlgebraicMatrix<f64>) -> Self {
        d.map(|x| da!(*x)).into()
    }
}

impl From<AlgebraicMatrix<f64>> for AlgebraicMatrix<DA> {
    fn from(d: AlgebraicMatrix<f64>) -> Self {
        (&d).into()
    }
}

impl From<&AlgebraicMatrix<u32>> for AlgebraicMatrix<DA> {
    fn from(d: &AlgebraicMatrix<u32>) -> Self {
        d.map(|x| da!(*x)).into()
    }
}

impl From<AlgebraicMatrix<u32>> for AlgebraicMatrix<DA> {
    fn from(d: AlgebraicMatrix<u32>) -> Self {
        (&d).into()
    }
}

impl<'a, T: Clone> From<ArrayView2<'a, T>> for AlgebraicMatrix<T> {
    fn from(v: ArrayView2<'a, T>) -> Self {
        AlgebraicMatrix(v.to_owned())
    }
}

// Add vector

impl_op_ex!(+ |a: &AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
    (&a.0 + &b.0).into()
});

impl_op_ex!(+= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| {
    a.0 += &b.0;
});

impl_op_ex_commutative!(+ |a: &AlgebraicMatrix<DA>, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<DA> {
    (&a.0 + &b.0).into()
});

impl_op_ex!(+= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<f64>| {
    a.0 = &a.0 + &b.0;
});

impl_op_ex!(+ |a: &AlgebraicMatrix<f64>, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<f64> {
    (&a.0 + &b.0).into()
});

impl_op_ex!(+= |a: &mut AlgebraicMatrix<f64>, b: &AlgebraicMatrix<f64>| {
    a.0 += &b.0;
});

// Add scalar

impl_op_ex_commutative!(+ |a: &AlgebraicMatrix<DA>, b: &DA| -> AlgebraicMatrix<DA> {
    (&a.0 + b.to_owned()).into()
});

impl_op_ex!(+= |a: &mut AlgebraicMatrix<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs += b;
    }
});

impl_op_ex_commutative!(+ |a: &AlgebraicMatrix<DA>, b: &f64| -> AlgebraicMatrix<DA> {
    (&a.0 + *b).into()
});

impl_op_ex!(+= |a: &mut AlgebraicMatrix<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs += b;
    }
});

impl_op_ex_commutative!(+ |a: &AlgebraicMatrix<f64>, b: &DA| -> AlgebraicMatrix<DA> {
    AlgebraicMatrix::<DA>::from(a) + b
});

impl_op_ex_commutative!(+ |a: &AlgebraicMatrix<f64>, b: &f64| -> AlgebraicMatrix<f64> {
    (&a.0 + *b).into()
});

impl_op_ex!(+= |a: &mut AlgebraicMatrix<f64>, b: &f64| {
    a.0 += *b
});

// Sub matrix

impl_op_ex!(
    -|a: &AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
        (&a.0 - &b.0).into()
    }
);

impl_op_ex!(-= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| {
    a.0 -= &b.0;
});

impl_op_ex!(
    -|a: &AlgebraicMatrix<DA>, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<DA> {
        (&a.0 - &b.0).into()
    }
);

impl_op_ex!(
    -|a: &AlgebraicMatrix<f64>, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
        (&a.0 - b.0.to_owned()).into()
    }
);

impl_op_ex!(
    -|a: &AlgebraicMatrix<f64>, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<f64> {
        (&a.0 - &b.0).into()
    }
);

impl_op_ex!(-= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<f64>| {
    for (lhs, rhs) in zip(&mut a.0, &b.0) {
        *lhs -= *rhs;
    }
});

// Sub scalar

impl_op_ex!(-|a: &AlgebraicMatrix<DA>, b: &DA| -> AlgebraicMatrix<DA> {
    (&a.0 - b.to_owned()).into()
});

impl_op_ex!(-|a: &DA, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
    (-(&b.0 - a.to_owned())).into()
});

impl_op_ex!(-= |a: &mut AlgebraicMatrix<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs -= b;
    }
});

impl_op_ex!(-|a: &AlgebraicMatrix<DA>, b: &f64| -> AlgebraicMatrix<DA> { (&a.0 - *b).into() });

impl_op_ex!(-|a: &f64, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> { (-(&b.0 - *a)).into() });

impl_op_ex!(-= |a: &mut AlgebraicMatrix<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs -= b;
    }
});

impl_op_ex!(-|a: &AlgebraicMatrix<f64>, b: &DA| -> AlgebraicMatrix<DA> {
    AlgebraicMatrix::<DA>::from(a) - b
});

impl_op_ex!(-|a: &DA, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<DA> {
    a - AlgebraicMatrix::<DA>::from(b)
});

impl_op_ex!(-|a: &AlgebraicMatrix<f64>, b: &f64| -> AlgebraicMatrix<f64> { (&a.0 - *b).into() });

impl_op_ex!(-|a: &f64, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<f64> { (-(&b.0 - *a)).into() });

impl_op_ex!(-= |a: &mut AlgebraicMatrix<f64>, b: &f64| {
    a.0 -= *b;
});

// Mul matrix

impl_op_ex!(
    *|a: &AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
        (&a.0 * &b.0).into()
    }
);

impl_op_ex!(*= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| {
    a.0 *= &b.0;
});

impl_op_ex_commutative!(*|a: &AlgebraicMatrix<DA>,
                          b: &AlgebraicMatrix<f64>|
 -> AlgebraicMatrix<DA> { (&a.0 * &b.0).into() });

impl_op_ex!(*= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<f64>| {
    for (lhs, rhs) in zip(&mut a.0, &b.0) {
        *lhs *= *rhs;
    }
});

// Mul scalar

impl_op_ex_commutative!(*|a: &AlgebraicMatrix<DA>, b: &DA| -> AlgebraicMatrix<DA> {
    (&a.0 * b.to_owned()).into()
});

impl_op_ex!(*= |a: &mut AlgebraicMatrix<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs *= b;
    }
});

impl_op_ex_commutative!(*|a: &AlgebraicMatrix<DA>, b: &f64| -> AlgebraicMatrix<DA> {
    (&a.0 * *b).into()
});

impl_op_ex!(*= |a: &mut AlgebraicMatrix<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs *= b;
    }
});

impl_op_ex_commutative!(*|a: &AlgebraicMatrix<f64>, b: &DA| -> AlgebraicMatrix<DA> {
    AlgebraicMatrix::<DA>::from(a) * b
});

impl_op_ex_commutative!(
    *|a: &AlgebraicMatrix<f64>, b: &f64| -> AlgebraicMatrix<f64> { (&a.0 * *b).into() }
);

impl_op_ex!(*= |a: &mut AlgebraicMatrix<f64>, b: &f64| {
    a.0 *= *b;
});

// Div matrix

impl_op_ex!(
    /|a: &AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
        (&a.0 / &b.0).into()
    }
);

impl_op_ex!(/= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<DA>| {
    a.0 /= &b.0;
});

impl_op_ex!(
    /|a: &AlgebraicMatrix<DA>, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<DA> {
        (&a.0 / &b.0).into()
    }
);

impl_op_ex!(
    /|a: &AlgebraicMatrix<f64>, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
        (&a.0 / b.0.to_owned()).into()
    }
);

impl_op_ex!(/= |a: &mut AlgebraicMatrix<DA>, b: &AlgebraicMatrix<f64>| {
    for (lhs, rhs) in zip(&mut a.0, &b.0) {
        *lhs /= *rhs;
    }
});

// Div scalar

impl_op_ex!(/|a: &AlgebraicMatrix<DA>, b: &DA| -> AlgebraicMatrix<DA> {
    (&a.0 / b.to_owned()).into()
});

impl_op_ex!(/|a: &DA, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
    (Array2::<DA>::ones([b.nrows(), b.ncols()]) * a.to_owned() / &b.0).into()
});

impl_op_ex!(/= |a: &mut AlgebraicMatrix<DA>, b: &DA| {
    for lhs in &mut a.0 {
        *lhs /= b;
    }
});

impl_op_ex!(/|a: &AlgebraicMatrix<DA>, b: &f64| -> AlgebraicMatrix<DA> { (&a.0 / *b).into() });

impl_op_ex!(/|a: &f64, b: &AlgebraicMatrix<DA>| -> AlgebraicMatrix<DA> {
    (Array2::<DA>::ones([b.nrows(), b.ncols()]) * *a / &b.0).into()
});

impl_op_ex!(/= |a: &mut AlgebraicMatrix<DA>, b: &f64| {
    for lhs in &mut a.0 {
        *lhs /= b;
    }
});

impl_op_ex!(/|a: &AlgebraicMatrix<f64>, b: &DA| -> AlgebraicMatrix<DA> {
    AlgebraicMatrix::<DA>::from(a) / b
});

impl_op_ex!(/|a: &DA, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<DA> {
    a / AlgebraicMatrix::<DA>::from(b)
});

impl_op_ex!(/|a: &AlgebraicMatrix<f64>, b: &f64| -> AlgebraicMatrix<f64> { (&a.0 / *b).into() });

impl_op_ex!(/|a: &f64, b: &AlgebraicMatrix<f64>| -> AlgebraicMatrix<f64> {
    (*a / &b.0).into()
});

impl_op_ex!(/= |a: &mut AlgebraicMatrix<f64>, b: &f64| {
    a.0 /= *b;
});

// Neg

impl<T> ops::Neg for AlgebraicMatrix<T>
where
    AlgebraicMatrix<T>: ops::MulAssign<f64>,
{
    type Output = AlgebraicMatrix<T>;
    fn neg(mut self) -> Self::Output {
        self *= -1.0;
        self
    }
}

impl<T> ops::Neg for &AlgebraicMatrix<T>
where
    T: Clone + ops::Mul<f64, Output = T>,
{
    type Output = AlgebraicMatrix<T>;
    fn neg(self) -> Self::Output {
        (&self.0 * -1.0).into()
    }
}
