use self::layer::Layer;

mod layer;

#[derive(Clone)]
pub struct NeuralNetwork {
    input_size: usize,
    layers: Vec<Layer>,
}
