use dace::*;

fn main() {
    DA::init(20, 1);

    let x = da!(1);

    // compute and print sin(x)^2
    let y1 = x.sin().sqr();
    println!("sin(x)^2\n{y1}");

    // compute and print cos(x)^2
    let y2 = x.cos().sqr();
    println!("cos(x)^2\n{y2}");

    // compute and print sin(x)^2+cos(x)^2
    let s = y1 + y2;
    println!("sin(x)^2+cos(x)^2\n{s}");
}
