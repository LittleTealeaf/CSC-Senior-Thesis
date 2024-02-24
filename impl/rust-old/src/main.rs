#![allow(dead_code)]

mod network;
mod utils;

use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::SystemTime,
};

use network::NeuralNetwork;

fn main() {
    let out_path_env = env::var("OUT_PATH").ok();

    if let Some(path) = &out_path_env {
        let _ = fs::remove_dir_all(path);
    }

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

    for (i, bootstrap) in bootstraps.into_iter().enumerate() {
        let data = bootstrap
            .into_iter()
            .filter_map(|bootstrap| data.get(bootstrap))
            .collect::<Vec<_>>();

        println!("Iteration {i}");

        let start = SystemTime::now();

        network.train(data, &0.1);

        let elapsed = start.elapsed().unwrap();
        times.push(elapsed.as_nanos());
    }

    if let Some(path) = &out_path_env {
        let out_path = PathBuf::from(path);

        fs::create_dir_all(&out_path).unwrap();

        {
            let times_file = out_path.join(Path::new("results.csv"));

            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(times_file)
                .unwrap();

            file.write_all("id,time".as_bytes()).unwrap();

            for (i, time) in times.into_iter().enumerate() {
                writeln!(&mut file, "{i},{time}").unwrap();
            }
        }

        {
            let network_file = out_path.join(Path::new("network"));

            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(network_file)
                .unwrap();

            network.write_to_file(&mut file).unwrap();
        }
    }
}
