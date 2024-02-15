#![allow(dead_code)]

mod network;

use std::time::SystemTime;

use network::NeuralNetwork;
use rand::thread_rng;

fn main() {
    let mut rng = thread_rng();

    let mut network = NeuralNetwork::from_str(include_str!("../../../data/network"));

    for _ in 0..1000 {
        let start = SystemTime::now();

        let elapsed = start.elapsed().unwrap();
    }
}
