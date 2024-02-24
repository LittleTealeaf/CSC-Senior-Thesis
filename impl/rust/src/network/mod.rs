use std::{
    fs::File,
    io::{Error, Write},
    ops::MulAssign,
};

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

    pub fn train<'a, I>(&mut self, data: I, alpha: &f64)
    where
        I: IntoIterator<Item = &'a Vec<f64>>,
        I::IntoIter: ExactSizeIterator + Sized + Send,
    {
        let iterator = data.into_iter();
        let count = iterator.len();

        let alpha = alpha / count as f64;

        let nudges = iterator
            .par_bridge()
            .map(|data| (self.back_propagate(&data[1..], &data[0], &alpha)))
            .reduce(Vec::new, |mut a, b| {
                if a.is_empty() {
                    b
                } else if b.is_empty() {
                    a
                } else {
                    for (i, layer) in b.into_iter().enumerate() {
                        a[i] += layer
                    }
                    a
                }
            });

        for (index, layer) in nudges.into_iter().enumerate() {
            self.layers[index] += layer * (1f64 / count as f64);
        }
    }

    fn back_propagate(&self, input: &[f64], expected: &f64, alpha: &f64) -> Vec<Layer> {
        struct Entry<'a> {
            inputs: Vec<f64>,
            outputs: Vec<f64>,
            layer: &'a Layer,
        }

        let mut entries = Vec::with_capacity(self.layers.len());
        let mut inputs = input.to_vec();

        for layer in &self.layers {
            let outputs = layer.feed_forward(inputs.clone());
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
            let (nudge, new_errors) = layer.back_propagate(&inputs, &outputs, &errors);
            errors = new_errors;
            nudges.push(nudge);
        }

        for nudge in &mut nudges {
            nudge.mul_assign(*alpha);
        }

        nudges.reverse();

        nudges
    }

    pub fn write_to_file(&self, file: &mut File) -> Result<(), Error> {
        for layer in &self.layers {
            write!(
                file,
                "{} {}\n{}\n",
                layer.input,
                layer.output,
                layer
                    .bias
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )?;
            for i in 0..layer.input {
                let start = layer.get_weights_index(i, 0);
                let end = layer.get_weights_index(i, layer.output);
                let items = &layer.weights[start..end];
                write!(
                    file,
                    "{}\n",
                    items
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )?;
            }
            write!(file, "\n")?;
        }
        Ok(())
    }
}
