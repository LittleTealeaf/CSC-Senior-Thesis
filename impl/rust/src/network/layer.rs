#[derive(Clone, Debug)]
pub struct Layer {
    pub input: usize,
    pub output: usize,
    pub bias: Vec<f64>,
    pub weights: Vec<Vec<f64>>,
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
            .map(|line| line.split(',').filter_map(|num| num.parse().ok()).collect())
            .collect();

        Some(Self {
            input,
            output,
            bias,
            weights,
        })
    }
}
