use std::{
    env,
    fs::{create_dir_all, remove_dir_all, File},
    io::Write,
    path::PathBuf,
    time::SystemTime,
};

use network::NeuralNetwork;

mod layer;
mod network;

fn main() {
    let out_env = env::var("OUT_PATH").ok();

    if let Some(out_path) = &out_env {
        let _ = remove_dir_all(out_path);
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
        let data = bootstrap.into_iter().map(|i| &data[i]).collect::<Vec<_>>();

        println!("Iteration {i}");

        let start = SystemTime::now();

        network.train(data, &0.1);

        let elapsed = start.elapsed().unwrap();

        times.push(elapsed.as_nanos());
    }

    if let Some(out_path) = &out_env {
        let base_path = PathBuf::from(out_path);

        create_dir_all(&base_path).unwrap();

        {
            let path = base_path.join("results.csv");

            let mut file = File::create(path).unwrap();

            writeln!(file, "id,time").unwrap();

            for (i, time) in times.into_iter().enumerate() {
                writeln!(file, "{i},{time}").unwrap();
            }
        }

        {
            let path = base_path.join("network");

            let mut file = File::create(path).unwrap();

            network.write_to_file(&mut file).unwrap();
        }
    }
}
