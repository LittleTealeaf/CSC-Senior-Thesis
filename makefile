# Rust
bin/rust: data $(wildcard impl/rust/src/*.rs) $(wildcard impl/rust/src/**/*.rs)
	mkdir bin 2> /dev/null || true
	cargo build -p rust-impl --release
	mv -u target/release/rust-impl bin/rust

out/rust: bin/rust
	OUT_PATH="out/rust" bin/rust

# Rust Tensorflow CPU
bin/rust-tf-cpu: data $(wildcard impl/rust-tf/src/*.rs)
	mkdir bin 2> /dev/null || true
	cargo build -p rust-tf --release
	mv -u target/release/rust-tf bin/rust-tf-cpu

out/rust-tf-cpu: bin/rust-tf-cpu
	OUT_PATH="out/rust-tf-cpu" bin/rust-tf-cpu

# Rust Tensorflow GPU
bin/rust-tf-gpu: data $(wildcard impl/rust-tf/src/*.rs)
	mkdir bin 2> /dev/null || true
	cargo build -p rust-tf --release --features gpu
	mv -u target/release/rust-tf bin/rust-tf-gpu

out/rust-tf-gpu: bin/rust-tf-gpu
	OUT_PATH="out/rust-tf-gpu" bin/rust-tf-gpu

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
bin/cuda: $(wildcard impl/cuda/*.cu)
	mkdir bin 2> /dev/null || true
	nvcc impl/cuda/main.cu -o bin/cuda

out/cuda: bin/cuda
	OUT_PATH="out/cuda" bin/cuda

# Misc

data: data.toml
	python3 data.py

clean:
	rm -r bin 2> /dev/null || true
	rm -r data 2> /dev/null || true
	rm -r out 2> /dev/null || true
	cargo clean

rmdata:
	rm -r data 2> /dev/null || true
