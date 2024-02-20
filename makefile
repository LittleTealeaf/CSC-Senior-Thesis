
# Rust Executable
impl/rust/target/release/rust-impl: data impl/rust/src
	cd impl/rust && cargo build --release

impl/rust/results.csv: impl/rust/target/release/rust-impl
	RESULTS_PATH="impl/rust/results.csv" impl/rust/target/release/rust-impl

# Tensorflow
impl/tensorflow/results.csv: impl/tensorflow/src data
	RESULTS_PATH="impl/tensorflow/results.csv" python3 impl/tensorflow/src/main.py

# CUDA
impl/cuda/executable: impl/cuda/src
	nvcc impl/cuda/src/main.cu -o impl/cuda/executable

impl/cuda/results.csv: data impl/cuda/executable
	RESULTS_PATH="impl/cuda/results.csv" ./impl/cuda/executable

# Misc.
data: data.toml
	python3 data.py

clean:
	rm -r data
