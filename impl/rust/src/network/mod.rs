use self::layer::Layer;

mod layer;

#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    input_size: usize,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn from_str(input: &str) -> Self {
        let layers = input
            .split("\n\n")
            .filter_map(Layer::from_str)
            .collect::<Vec<_>>();

        Self {
            input_size: layers[0].input,
            layers,
        }
    }
}
