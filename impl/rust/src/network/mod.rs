use self::layer::Layer;
use rayon::prelude::*;

mod layer;
mod activation;

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

    pub fn train<'a>(&mut self, data: impl IntoParallelIterator<Item = &'a Vec<f64>>) {
        data.into_par_iter().map(|_data| 0f64).collect::<Vec<_>>();
    }
}
