use tensorflow::{
    ops::{self, mat_mul, relu, Placeholder},
    train::{GradientDescentOptimizer, MinimizeOptions, Optimizer},
    DataType, Graph, Operation, Scope, Session, SessionOptions, SessionRunArgs, Status, Tensor,
    Variable,
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

    let network = load_network(include_str!("../../../data/network"));

    let mut scope = Scope::new_root_scope();

    let input = Placeholder::new()
        .dtype(DataType::Double)
        .shape([1u64, (data[0].len() - 1) as u64])
        .build(&mut scope.with_op_name("input"))
        .unwrap();

    let label = Placeholder::new()
        .dtype(DataType::Double)
        .shape([1, 1])
        .build(&mut scope.with_op_name("label"))
        .unwrap();

    let mut layer_in = input.clone();
    let mut variables = Vec::new();

    for (index, (bias, weights)) in network.into_iter().enumerate() {
        let mut scope = scope.new_sub_scope(&format!("layer_{index}"));
        let w = Variable::builder()
            .const_initial_tensor(&weights)
            .build(&mut scope.with_op_name("w"))
            .unwrap();

        let b = Variable::builder()
            .const_initial_tensor(&bias)
            .build(&mut scope.with_op_name("b"))
            .unwrap();

        layer_in = relu(
            ops::add(
                b.output().clone(),
                ops::mat_mul(layer_in, w.output().clone(), &mut scope).unwrap(),
                &mut scope,
            )
            .unwrap(),
            &mut scope,
        )
        .unwrap();

        variables.extend([w, b]);
    }

    let error = ops::sub(layer_in.clone(), label.clone(), &mut scope).unwrap();
    let error_squared = ops::mul(error.clone(), error, &mut scope).unwrap();
    let optimizer = GradientDescentOptimizer::new(ops::constant(0.1f64, &mut scope).unwrap());

    let (variables, optimizer) = optimizer
        .minimize(
            &mut scope,
            error_squared.clone().into(),
            MinimizeOptions::default().with_variables(&variables),
        )
        .unwrap();

    let mut options = SessionOptions::default();
    options.set_target(&optimizer.name().unwrap()).unwrap();

    let session = Session::new(&options, &scope.graph()).unwrap();

    for bootstrap in bootstraps {
        let input_tensor = Tensor::new(&[bootstrap.len() as u64, data[0].len() as u64 - 1])
            .with_values(
                &(bootstrap
                    .iter()
                    .flat_map(|i| data[*i][1..].to_vec().clone())
                    .collect::<Vec<_>>()),
            )
            .unwrap();

        let label_tensor = Tensor::new(&[bootstrap.len() as u64, 1])
            .with_values(&(bootstrap.iter().map(|i| data[*i][0]).collect::<Vec<_>>()))
            .unwrap();

        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&input, 0, &input_tensor);
        run_args.add_feed(&label, 0, &label_tensor);

        session.run(&mut run_args).unwrap();
    }
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

        let weights = Tensor::new(&[input_size, output_size])
            .with_values(&weights_values)
            .unwrap();

        (bias, weights)
    })
}
