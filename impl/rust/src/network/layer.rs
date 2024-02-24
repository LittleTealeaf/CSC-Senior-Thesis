use std::ops::{Add, AddAssign, Mul, MulAssign};

use super::activation::Activation;

#[derive(Clone, Debug)]
pub struct Layer {
    pub input: usize,
    pub output: usize,
    pub bias: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Layer {
    #[inline(always)]
    pub fn from_str(input: &str) -> Option<Self> {
        let mut lines = input.lines();
        let mut dims = lines.next()?.split(' ');

        let input = dims.next()?.parse().ok()?;
        let output = dims.next()?.parse().ok()?;

        let bias = lines
            .next()?
            .split(',')
            .filter_map(|num| num.parse().ok())
            .collect();
        let weights = lines
            .flat_map(|line| line.split(',').filter_map(|num| num.parse().ok()))
            .collect();

        Some(Self {
            input,
            output,
            bias,
            weights,
        })
    }

    #[inline(always)]
    pub fn copy_size(&self) -> Self {
        Self {
            input: self.input,
            output: self.output,
            weights: vec![0f64; self.input * self.output],
            bias: vec![0f64; self.output],
        }
    }

    pub fn feed_forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut output = self.bias.clone();
        for j in 0..self.output {
            (0..self.input).for_each(|i| {
                output[j] += input[i] * self.weights[self.get_weights_index(i, j)];
            });
            output[j] = output[j].activation();
        }
        output
    }

    #[inline(always)]
    pub fn get_weights_index(&self, input: usize, output: usize) -> usize {
        input * self.output + output
    }

    pub fn back_propagate(
        &self,
        input: &[f64],
        output: &[f64],
        errors: &[f64],
    ) -> (Layer, Vec<f64>) {
        let mut nudge = self.copy_size();
        let mut new_errors = vec![0f64; self.input];
        for j in 0..self.output {
            let error = output[j].activation_derivative() * errors[j];
            (0..self.input).for_each(|i| {
                let index = self.get_weights_index(i, j);
                new_errors[j] += error * self.weights[index];
                nudge.weights[index] += input[i] * error;
            });
            nudge.bias[j] += error;
        }
        (nudge, new_errors)
    }
}

impl Add for Layer {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign for Layer {
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.output == rhs.output);
        assert!(self.input == rhs.input);

        for i in 0..self.output {
            self.bias[i] += rhs.bias[i];

            for j in 0..self.input {
                let index = self.get_weights_index(j, i);
                self.weights[index] += rhs.weights[index];
            }
        }
    }
}

impl Mul<f64> for Layer {
    type Output = Self;
    fn mul(mut self, rhs: f64) -> Self::Output {
        self *= rhs;
        self
    }
}

impl MulAssign<f64> for Layer {
    fn mul_assign(&mut self, rhs: f64) {
        for o in 0..self.output {
            self.bias[o] *= rhs;
            for i in 0..self.output {
                let index = self.get_weights_index(o, i);
                self.weights[index] *= rhs;
            }
        }
    }
}
