use tensorflow::{
    ops::{self, mat_mul, relu, Placeholder},
    train::Optimizer,
    DataType, Graph, Scope, Tensor,
};

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
    let back_prop = back_propagation(network, &mut Scope::new_root_scope());
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
        let bias = Tensor::new(&[1, output_size])
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

fn back_propagation(
    layers: impl IntoIterator<Item = (Tensor<f64>, Tensor<f64>)>,
    scope: &mut Scope,
) {
    let input = Placeholder::new()
        .dtype(DataType::Double)
        .shape([1u64, 10u64])
        .build(&mut scope.with_op_name("input"))
        .unwrap();

    let label = Placeholder::new()
        .dtype(DataType::Double)
        .shape([1, 1])
        .build(&mut scope.with_op_name("label"))
        .unwrap();

    let feed_forward = layers.into_iter().fold(input, |prev, (bias, weight)| {
        let mul = mat_mul(input, weight, scope).unwrap();
        let add = tensorflow::ops::add(mul, bias, scope).unwrap();
        let output = relu(add, scope).unwrap();
        output
    });

    let label = ops::sub(feed_forward, label, scope).unwrap();
}
