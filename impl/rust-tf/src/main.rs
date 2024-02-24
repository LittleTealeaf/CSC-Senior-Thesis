extern crate tensorflow;

fn main() {
    let data: Vec<Vec<f64>> = include_str!("../../../data/data.csv")
        .lines()
        .map(|line| line.split(',').filter_map(|n| n.parse().ok()).collect())
        .collect();

    let bootstraps: Vec<Vec<usize>> = include_str!("../../../data/bootstraps.csv")
        .lines()
        .map(|line| line.split(',').filter_map(|n| n.parse().ok()).collect())
        .collect();
}



