#![allow(dead_code)]

mod network;
mod utils;

use std::{env, fs::OpenOptions, io::Write, time::SystemTime};

use network::NeuralNetwork;

fn main() {
    let data: Vec<Vec<f64>> = include_str!("../../../data/data.csv")
        .lines()
        .map(|line| line.split(',').filter_map(|n| n.parse().ok()).collect())
        .collect();

    let bootstraps: Vec<Vec<usize>> = include_str!("../../../data/bootstraps.csv")
        .lines()
        .map(|line| line.split(',').filter_map(|n| n.parse().ok()).collect())
        .collect();

    let mut network = NeuralNetwork::from_str(include_str!("../../../data/network"));

    let mut times = Vec::new();

    for bootstrap in bootstraps {
        let data = bootstrap
            .into_iter()
            .filter_map(|bootstrap| data.get(bootstrap))
            .collect::<Vec<_>>();

        let start = SystemTime::now();

        network.train(data, &0.1);

        let elapsed = start.elapsed().unwrap();
        times.push(elapsed.as_nanos());
    }

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(env::var("RESULTS_PATH").unwrap())
        .unwrap();

    file.write_all("id,time".as_bytes()).unwrap();

    for (i, time) in times.into_iter().enumerate() {
        writeln!(&mut file, "{i},{time}").unwrap();
    }
}
