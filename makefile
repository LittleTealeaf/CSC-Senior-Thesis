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
target/cuda: $(wildcard impl/cuda/*.cu)
	mkdir target 2> /dev/null; nvcc impl/cuda/main.cu -o target/cuda

out/cuda: target/cuda
	OUT_PATH="out/cuda" target/cuda

# Misc.
data: data.toml
	python3 data.py

clean:
	rm -r data 2> /dev/null; rm -r out 2> /dev/null; cargo clean
