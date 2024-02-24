# Rust Executable
target/release/rust-impl: data $(wildcard impl/rust/src/**.rs)
	cargo build -p rust-impl --release

out/rust: target/release/rust-impl
	OUT_PATH="out/rust" target/release/rust-impl


# Tensorflow
out/tensorflow: impl/tensorflow/main.py data
	OUT_PATH="out/tensorflow" PROJECT_ROOT="true" python3 impl/tensorflow/main.py

# Tensorflow CPU

out/tensorflow-cpu: impl/tensorflow-cpu/main.py data
	OUT_PATH="out/tensorflow-cpu" PROJECT_ROOT="true" python3 impl/tensorflow-cpu/main.py

# CUDA
impl/cuda/executable: impl/cuda/src
	nvcc impl/cuda/src/main.cu -o impl/cuda/executable

impl/cuda/results.csv: data impl/cuda/executable
	RESULTS_PATH="impl/cuda/results.csv" ./impl/cuda/executable && echo "id,time" >> impl/cuda/results.csv

# Misc.
data: data.toml
	python3 data.py

clean:
	rm -r data && rm -r out
