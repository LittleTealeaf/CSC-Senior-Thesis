use tensorflow::Tensor;

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

    let network = load_network(include_str!("../../../data/network")).collect::<Vec<_>>();
}

fn load_network(string: &str) -> impl Iterator<Item = (Tensor<f64>, Tensor<f64>)> + '_ {
    let layers = string.trim().split("\n\n");

    layers.map(|layer| {
        let mut lines = layer.lines();

        let mut dims = lines
            .next()
            .unwrap()
            .split(' ')
            .map(|i| i.parse::<u64>().unwrap());
        let input_size = dims.next().unwrap();
        let output_size = dims.next().unwrap();

        let bias_values = lines
            .next()
            .unwrap()
            .split(',')
            .map(|i| i.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        let bias = Tensor::new(&[output_size])
            .with_values(&bias_values)
            .unwrap();

        let weights_values = lines
            .flat_map(|line| line.split(',').map(|i| i.parse::<f64>().unwrap()))
            .collect::<Vec<_>>();

        let weights = Tensor::new(&[output_size, input_size])
            .with_values(&weights_values)
            .unwrap();

        (bias, weights)
    })
}
