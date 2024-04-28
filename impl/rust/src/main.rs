use std::{
    env, fs::File, io::{Read, Write}, path::PathBuf, time::SystemTime
};

use network::NeuralNetwork;

mod layer;
mod network;

fn read_from_file(path: String) -> String {
    let mut file = File::open(path.clone()).unwrap();
    let mut s = String::new();
    println!("{}", path);
    file.read_to_string(&mut s).unwrap();
    return s;
}

fn main() {
    let out_env = env::var("OUT_PATH").ok();

    let data: Vec<Vec<f64>> = read_from_file(env::var("DATASET_PATH").unwrap())
        .lines()
        .map(|line| line.split(',').filter_map(|n| n.parse().ok()).collect())
        .collect();

    println!("Data Loaded");

    let bootstraps: Vec<Vec<usize>> = read_from_file(env::var("BOOTSTRAP_PATH").unwrap())
        .lines()
        .map(|line| line.split(',').filter_map(|n| n.parse().ok()).collect())
        .collect();

    println!("Bootstraps Loaded");

    let mut network = NeuralNetwork::from_str(&read_from_file(env::var("NETWORK_PATH").unwrap()));

    println!("Network Loaded");

    let mut times = Vec::new();

    for (_sample_n, bootstrap) in bootstraps.into_iter().enumerate() {

        let data = bootstrap.into_iter().map(|i| &data[i]).collect::<Vec<_>>();

        let start = SystemTime::now();

        network.train(data, &0.1);

        let elapsed = start.elapsed().unwrap();

        times.push(elapsed.as_nanos());
    }

    if let Some(out_path) = &out_env {
        let path = PathBuf::from(out_path);
        let mut file = File::create(path).unwrap();

        for (i, time) in times.into_iter().enumerate() {
            writeln!(file, "{i},{time}").unwrap();
        }
    }
}
