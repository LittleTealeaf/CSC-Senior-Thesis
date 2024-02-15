use super::activation::Activation;

#[derive(Clone, Debug)]
pub struct Layer {
    pub input: usize,
    pub output: usize,
    pub bias: Vec<f64>,
    pub weights: Vec<f64>,
}

impl Layer {
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
    fn get_weights_index(&self, input: usize, output: usize) -> usize {
        input * self.output + output
    }
}
