use std::{
    fs::File,
    io::{Error, Write},
    ops::{AddAssign, Mul},
};

trait Relu {
    fn relu(self) -> Self;
    fn relu_der(self) -> Self;
}

impl Relu for f64 {
    fn relu(self) -> Self {
        self.max(0f64)
    }

    fn relu_der(self) -> Self {
        if self >= 0f64 {
            1f64
        } else {
            0f64
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    input: usize,
    output: usize,
    bias: Vec<f64>,
    weights: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct LayerBackPropagate {
    pub nudges: Layer,
    pub errors: Vec<f64>,
}

impl Layer {
    pub fn from_str(input: &str) -> Option<Self> {
        let mut lines = input.trim().lines();
        let mut dims = lines.next()?.split(' ');

        let input = dims.next()?.parse().unwrap();
        let output = dims.next()?.parse().unwrap();

        let bias = lines
            .next()?
            .split(',')
            .map(|n| n.parse().unwrap())
            .collect();

        let weights = lines
            .flat_map(|line| line.split(',').map(|num| num.parse().unwrap()))
            .collect();

        Some(Self {
            input,
            output,
            bias,
            weights,
        })
    }

    #[inline(always)]
    fn get_weight_index(&self, input: usize, output: usize) -> usize {
        input * self.output + output
    }

    pub fn feed_forward(&self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(input.len(), self.input);
        let mut output = self.bias.clone();
        for j in 0..self.output {
            (0..self.input)
                .for_each(|i| output[j] += input[i] * self.weights[self.get_weight_index(i, j)]);
            output[j] = output[j].relu();
        }
        output
    }

    fn copy_size(&self) -> Self {
        Self {
            input: self.input,
            output: self.output,
            weights: vec![0f64; self.input * self.output],
            bias: vec![0f64; self.output],
        }
    }

    pub fn back_propagate(
        &self,
        input: &[f64],
        output: &[f64],
        forward_errors: &[f64],
    ) -> LayerBackPropagate {
        let mut nudges = self.copy_size();
        let mut errors = vec![0f64; self.input];

        for j in 0..self.output {
            let error = output[j].relu_der() * forward_errors[j];
            (0..self.input).for_each(|i| {
                let index = self.get_weight_index(i, j);
                errors[i] += error * self.weights[index];
                nudges.weights[index] += input[i] * error;
            });
            nudges.bias[j] += error;
        }

        LayerBackPropagate { nudges, errors }
    }

    pub fn write_to_file(&self, file: &mut File) -> Result<(), Error> {
        write!(
            file,
            "{} {}\n{}\n",
            self.input,
            self.output,
            self.bias
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )?;

        for i in 0..self.input {
            let start = self.get_weight_index(i, 0);
            let end = self.get_weight_index(i, self.output);
            let items = &self.weights[start..end];

            writeln!(
                file,
                "{}",
                items
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            )?;
        }
        writeln!(file)?;

        Ok(())
    }
}

impl AddAssign for Layer {
    fn add_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.output, rhs.output);
        debug_assert_eq!(self.input, rhs.input);

        for i in 0..self.output {
            self.bias[i] += rhs.bias[i];
            for j in 0..self.input {
                let index = self.get_weight_index(j, i);
                self.weights[index] += rhs.weights[index];
            }
        }
    }
}

impl Mul<f64> for Layer {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        for b in &mut self.bias {
            *b *= rhs;
        }
        for w in &mut self.weights {
            *w *= rhs;
        }

        self
    }
}
