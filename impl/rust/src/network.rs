use std::{fs::File, io::Error};

use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::layer::Layer;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn from_str(input: &str) -> Self {
        Self {
            layers: input
                .trim()
                .split("\n\n")
                .filter_map(Layer::from_str)
                .collect::<Vec<_>>(),
        }
    }

    pub fn train<'a, I>(&mut self, data: I, alpha: &f64)
    where
        I: IntoIterator<Item = &'a Vec<f64>>,
        I::IntoIter: ExactSizeIterator + Sized + Send,
    {
        let iterator = data.into_iter();
        let count = iterator.len();

        let nudges = iterator
            .par_bridge()
            .map(|data| {
                self.back_propagate(&data[1..], &data[0])
                    .into_iter()
                    .map(|layer| layer * (alpha / count as f64))
                    .collect()
            })
            .reduce(Vec::new, |mut a, b| {
                if a.is_empty() {
                    b
                } else if b.is_empty() {
                    a
                } else {
                    for (i, layer) in b.into_iter().enumerate() {
                        a[i] += layer;
                    }
                    a
                }
            });

        for (index, layer) in nudges.into_iter().enumerate() {
            self.layers[index] += layer;
        }
    }

    fn back_propagate(&self, input: &[f64], expected: &f64) -> Vec<Layer> {
        struct Entry<'a> {
            inputs: Vec<f64>,
            outputs: Vec<f64>,
            layer: &'a Layer,
        }

        let mut entries = Vec::with_capacity(self.layers.len());
        let mut inputs = input.to_vec();

        for layer in &self.layers {
            let outputs = layer.feed_forward(&inputs);
            entries.push(Entry {
                inputs,
                outputs: outputs.clone(),
                layer,
            });
            inputs = outputs;
        }

        let mut nudges = Vec::new();

        let mut errors = vec![*expected];

        while let Some(Entry {
            inputs,
            outputs,
            layer,
        }) = entries.pop()
        {
            let bp = layer.back_propagate(&inputs, &outputs, &errors);
            errors = bp.errors;
            nudges.push(bp.nudges);
        }

        nudges.reverse();
        nudges
    }

    pub fn write_to_file(&self, file: &mut File) -> Result<(), Error> {
        for layer in &self.layers {
            layer.write_to_file(file)?;
        }
        Ok(())
    }
}
