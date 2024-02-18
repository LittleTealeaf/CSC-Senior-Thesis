use self::layer::Layer;
use rayon::prelude::*;

mod activation;
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

    pub fn train<'a, I>(&mut self, data: I)
    where
        I: IntoParallelIterator<Item = &'a Vec<f64>>,
    {
        let nudges = data
            .into_par_iter()
            .map(|data| self.back_propagate(&data[1..], &data[0]))
            .collect::<Vec<_>>();
    }

    fn back_propagate(&self, input: &[f64], expected: &f64) -> Vec<Layer> {
        todo!()
    }
}
