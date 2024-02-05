use std::time::SystemTime;

use rand::{thread_rng, Rng};

fn main() {
    let mut rng = thread_rng();

    for _ in 0..1000 {
        let start = SystemTime::now();

        for _ in 0..1_000_000 {
            let _i: i64 = rng.gen();
        }

        let elapsed = start.elapsed().unwrap();

        println!("{}", elapsed.as_nanos());
    }
}
