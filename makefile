# Rust Executable
target/release/rust-impl: data $(wildcard impl/rust/src/*.rs) $(wildcard impl/rust/src/**/*.rs)
	cargo build -p rust-impl --release

out/rust: target/release/rust-impl
	OUT_PATH="out/rust" target/release/rust-impl

# Rust Tensorflow CPU
target/release/rust-tensorflow-cpu: data $(wildcard impl/rust-tensorflow/src/*.rs)
	cargo build -p rust-tensorflow --release && mv -u target/release/rust-tensorflow target/release/rust-tensorflow-cpu

out/rust-tensorflow-cpu: target/release/rust-tensorflow-cpu
	OUT_PATH="out/rust-tensorflow-cpu" target/release/rust-tensorflow-cpu

# Rust Tensorflow GPU
target/release/rust-tensorflow-gpu: data $(wildcard impl/rust-tensorflow/src/*.rs)
	cargo build -p rust-tensorflow --release --features gpu && mv -u target/release/rust-tensorflow target/release/rust-tensorflow-gpu

out/rust-tensorflow-gpu: target/release/rust-tensorflow-gpu
	OUT_PATH="out/rust-tensorflow-gpu" target/release/rust-tensorflow-gpu

# Tensorflow GPU
out/tensorflow-gpu: impl/tensorflow/main.py data
	OUT_PATH="out/tensorflow-gpu" PROJECT_ROOT="true" python3 impl/tensorflow/main.py

# Tensorflow CPU
out/tensorflow-cpu: impl/tensorflow/main.py data
	OUT_PATH="out/tensorflow-cpu" PROJECT_ROOT="true" CUDA_VISIBLE_DEVICES='-1' python3 impl/tensorflow/main.py

# CUDA
impl/cuda/executable: impl/cuda/src
	nvcc impl/cuda/src/main.cu -o impl/cuda/executable

impl/cuda/results.csv: data impl/cuda/executable
	RESULTS_PATH="impl/cuda/results.csv" ./impl/cuda/executable && echo "id,time" >> impl/cuda/results.csv

# Misc.
data: data.toml
	python3 data.py

clean:
	rm -r data; rm -r out; cargo clean
