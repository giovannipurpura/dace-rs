use dace::*;

fn main() {
    DA::init(10, 3);

    // initialize cyl
    let cyl = darray![100.0 + da!(1), da!(2) * PI / 180.0, da!(3)];

    // initialize cart and compute transformation
    let cart = darray![
        &cyl[0] * cyl[1].cos(),
        &cyl[0] * cyl[1].sin(),
        cyl[2].clone(),
    ];

    // subtract constant part to build DirMap
    let dir_map = &cart - cart.cons();

    println!("Direct map: from cylindric to cartesian (DirMap)\n{dir_map}\n\n");

    // Invert DirMap to obtain InvMap
    let inv_map = dir_map.invert().unwrap();

    println!("Inverse map: from cartesian to cylindric (InvMap)\n{inv_map}\n\n");

    // Verification
    println!(
        "Concatenate the direct and inverse map: (DirMap) o (InvMap) = DirMap(InvMap) = I\n\n"
    );
    println!("{}", dir_map.eval(&inv_map));
}
