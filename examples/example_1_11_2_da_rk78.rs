use dace::*;
use ndarray::{concatenate, s, Axis};
use ndarray_linalg::Norm;

const MU: f64 = 398600.0; // km^3/s^2

fn rk78(
    y0: &AlgebraicVector<DA>,
    x0: f64,
    x1: f64,
    f: fn(&AlgebraicVector<DA>, f64) -> AlgebraicVector<DA>,
) -> AlgebraicVector<DA> {
    let mut y0 = y0.to_owned();

    let n = y0.len();

    const H0: f64 = 0.001;
    const HS: f64 = 0.1;
    const H1: f64 = 100.0;
    const EPS: f64 = 1.0E-12;
    const BS: f64 = 20.0 * EPS;

    let mut z = AlgebraicMatrix::zeros((n, 16));
    let mut y1 = AlgebraicVector::zeros(n);

    let mut vihmax: f64 = 0.0;

    const HSQR: f64 = 1.0 / 9.0;

    const A: [f64; 13] = [
        0.0,
        1.0 / 18.0,
        1.0 / 12.0,
        1.0 / 8.0,
        5.0 / 16.0,
        3.0 / 8.0,
        59.0 / 400.0,
        93.0 / 200.0,
        5490023248.0 / 9719169821.0,
        13.0 / 20.0,
        1201146811.0 / 1299019798.0,
        1.0,
        1.0,
    ];
    const B: [[f64; 12]; 13] = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            1.0 / 18.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.0 / 48.0,
            1.0 / 16.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            1.0 / 32.0,
            0.0,
            3.0 / 32.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            5.0 / 16.0,
            0.0,
            -75.0 / 64.0,
            75.0 / 64.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            3.0 / 80.0,
            0.0,
            0.0,
            3.0 / 16.0,
            3.0 / 20.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            29443841.0 / 614563906.0,
            0.0,
            0.0,
            77736538.0 / 692538347.0,
            -28693883.0 / 1125000000.0,
            23124283.0 / 1800000000.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            16016141.0 / 946692911.0,
            0.0,
            0.0,
            61564180.0 / 158732637.0,
            22789713.0 / 633445777.0,
            545815736.0 / 2771057229.0,
            -180193667.0 / 1043307555.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            39632708.0 / 573591083.0,
            0.0,
            0.0,
            -433636366.0 / 683701615.0,
            -421739975.0 / 2616292301.0,
            100302831.0 / 723423059.0,
            790204164.0 / 839813087.0,
            800635310.0 / 3783071287.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            246121993.0 / 1340847787.0,
            0.0,
            0.0,
            -37695042795.0 / 15268766246.0,
            -309121744.0 / 1061227803.0,
            -12992083.0 / 490766935.0,
            6005943493.0 / 2108947869.0,
            393006217.0 / 1396673457.0,
            123872331.0 / 1001029789.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -1028468189.0 / 846180014.0,
            0.0,
            0.0,
            8478235783.0 / 508512852.0,
            1311729495.0 / 1432422823.0,
            -10304129995.0 / 1701304382.0,
            -48777925059.0 / 3047939560.0,
            15336726248.0 / 1032824649.0,
            -45442868181.0 / 3398467696.0,
            3065993473.0 / 597172653.0,
            0.0,
            0.0,
        ],
        [
            185892177.0 / 718116043.0,
            0.0,
            0.0,
            -3185094517.0 / 667107341.0,
            -477755414.0 / 1098053517.0,
            -703635378.0 / 230739211.0,
            5731566787.0 / 1027545527.0,
            5232866602.0 / 850066563.0,
            -4093664535.0 / 808688257.0,
            3962137247.0 / 1805957418.0,
            65686358.0 / 487910083.0,
            0.0,
        ],
        [
            403863854.0 / 491063109.0,
            0.0,
            0.0,
            -5068492393.0 / 434740067.0,
            -411421997.0 / 543043805.0,
            652783627.0 / 914296604.0,
            11173962825.0 / 925320556.0,
            -13158990841.0 / 6184727034.0,
            3936647629.0 / 1978049680.0,
            -160528059.0 / 685178525.0,
            248638103.0 / 1413531060.0,
            0.0,
        ],
    ];
    const C: [f64; 13] = [
        14005451.0 / 335480064.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -59238493.0 / 1068277825.0,
        181606767.0 / 758867731.0,
        561292985.0 / 797845732.0,
        -1041891430.0 / 1371343529.0,
        760417239.0 / 1151165299.0,
        118820643.0 / 751138087.0,
        -528747749.0 / 2220607170.0,
        1.0 / 4.0,
    ];
    const D: [f64; 13] = [
        13451932.0 / 455176623.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -808719846.0 / 976000145.0,
        1757004468.0 / 5645159321.0,
        656045339.0 / 265891186.0,
        -3867574721.0 / 1518517206.0,
        465885868.0 / 322736535.0,
        53011238.0 / 667516719.0,
        2.0 / 45.0,
        0.0,
    ];

    z.column_mut(0).assign(&y0);

    let mut h = HS.abs();
    let hh0 = H0.abs();
    let hh1 = H1.abs();
    let mut x = x0;
    let mut rfnorm = 0.0;

    while x != x1 {
        // compute new stepsize
        if rfnorm != 0.0 {
            h *= (HSQR * (EPS / rfnorm).ln()).exp().min(4.0);
        }

        if h.abs() > hh1.abs() {
            h = hh1;
        } else if h.abs() < hh0.abs() * 0.99 {
            h = hh0;
            println!("--- WARNING, MINIMUM STEPSIZE REACHED IN RK");
        }

        if (x + h - x1) * h > 0.0 {
            h = x1 - x;
        }

        for j in 0..13 {
            for i in 0..n {
                y0[i].set_zero();
                // EVALUATE RHS AT 13 POINTS
                for k in 0..j {
                    y0[i] += &z[[i, k + 3]] * B[j][k];
                }

                y0[i] = h * &y0[i] + &z[[i, 0]];
            }

            y1 = f(&y0, x + h * A[j]);

            z.column_mut(j + 3).assign(&y1);
        }

        for i in 0..n {
            z[[i, 1]].set_zero();
            z[[i, 2]].set_zero();
            // EXECUTE 7TH,8TH ORDER STEPS
            for j in 0..13 {
                z[[i, 1]] = &z[[i, 1]] + &z[[i, j + 3]] * D[j];
                z[[i, 2]] = &z[[i, 2]] + &z[[i, j + 3]] * C[j];
            }
            y1[i] = (&z[[i, 2]] - &z[[i, 1]]) * h;
            z[[i, 2]] = &z[[i, 2]] * h + &z[[i, 0]];
        }

        let y1_cons = y1.cons();

        // ESTIMATE ERROR AND DECIDE ABOUT BACKSTEP
        rfnorm = y1_cons.norm_max();
        if rfnorm > BS && (h / H0).abs() > 1.2 {
            h /= 3.0;
            rfnorm = 0.0;
        } else {
            let zc = z.column(2).to_owned();
            z.column_mut(0).assign(&zc);
            x += h;
            vihmax = vihmax.max(h);
        }
    }

    y1.assign(&z.column(0));
    y1
}

fn tbp(x: &AlgebraicVector<DA>, _t: f64) -> AlgebraicVector<DA> {
    let pos = x.slice(s![0..3]);
    let vel = x.slice(s![3..6]);

    let r = AlgebraicVector::from(pos).vnorm();
    let acc = -&pos * MU / r.pow(3);

    concatenate![Axis(0), vel, acc].into()
}

fn main() {
    DA::init(3, 6);

    // Set initial conditions
    let ecc = 0.5;

    let mut x0 = AlgebraicVector::identity(6);
    x0[0] += 6678.0; // 300 km altitude
    x0[4] += (MU / 6678.0 * (1.0 + ecc)).sqrt();

    // integrate for half the orbital period
    let a = 6678.0 / (1.0 - ecc);
    let xf = rk78(&x0, 0.0, PI * (a.powi(3) / MU).sqrt(), tbp);

    println!("Initial conditions:\n{}\n", x0);
    println!("Final conditions:\n{}\n", xf);
    println!("Initial conditions (cons. part):\n{}\n", x0.cons());
    println!("Final conditions: (cons. part)\n{}\n", xf.cons());

    // Evaluate for a displaced initial condition
    let delta_x0 = darray![1.0, -1.0, 0.0, 0.0, 0.0, 0.0]; // km

    println!("Displaced Initial condition:\n{}\n", x0.cons() + &delta_x0);
    println!("Displaced Final condition:\n{}\n", xf.eval(&delta_x0));
}
