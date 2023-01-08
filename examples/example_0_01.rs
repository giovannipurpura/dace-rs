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
