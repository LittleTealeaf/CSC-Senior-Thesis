# Rust
build/rust: data $(wildcard impl/rust/src/*.rs) $(wildcard impl/rust/src/**/*.rs)
	mkdir build 2> /dev/null || true
	cargo build -p rust-impl --release
	mv -u target/release/rust-impl build/rust

out/rust: build/rust
	OUT_PATH="out/rust" build/rust

# Rust Tensorflow CPU
build/rust-tf-cpu: data $(wildcard impl/rust-tf/src/*.rs)
	mkdir build 2> /dev/null || true
	cargo build -p rust-tf --release
	mv -u target/release/rust-tf build/rust-tf-cpu

out/rust-tf-cpu: build/rust-tf-cpu
	OUT_PATH="out/rust-tf-cpu" build/rust-tf-cpu

# Rust Tensorflow GPU
build/rust-tf-gpu: data $(wildcard impl/rust-tf/src/*.rs)
	mkdir build 2> /dev/null || true
	cargo build -p rust-tf --release --features gpu
	mv -u target/release/rust-tf build/rust-tf-gpu

out/rust-tf-gpu: build/rust-tf-cpu
	OUT_PATH="out/rust-tf-gpu" build/rust-tf-gpu

# Python Tensorflow GPU
out/python-tf-gpu: impl/python-tf/main.py data
	OUT_PATH="out/python-tf-gpu" PROJECT_ROOT="true" python3 impl/python-tf/main.py

# Python Tensorflow CPU
out/python-tf-cpu: impl/python-tf/main.py data
	OUT_PATH="out/python-tf-cpu" PROJECT_ROOT="true" CUDA_VISIBLE_DEVICES='-1' python3 impl/python-tf/main.py

# Python Numpy
out/python-np: impl/python-np/main.py data
	OUT_PATH="out/python-np" PROJECT_ROOT="true" python3 impl/python-np/main.py

# CUDA
build/cuda: $(wildcard impl/cuda/*.cu)
	mkdir build 2> /dev/null || true
	nvcc impl/cuda/main.cu -o build/cuda

out/cuda: target/cuda
	OUT_PATH="out/cuda" target/cuda

# Misc

data: data.toml
	python3 data.py

clean:
	rm -r build 2> /dev/null || true
	rm -r data 2> /dev/null || true
	rm -r out 2> /dev/null || true
	cargo clean
