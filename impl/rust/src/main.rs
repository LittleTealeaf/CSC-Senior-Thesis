use std::time::SystemTime;

fn main() {
    let start = SystemTime::now();
    println!("Hello");
    let elapsed = start.elapsed().unwrap();

    println!("{}", elapsed.as_nanos());
}
