data: data.toml
	python3 data.py

clean:
	rm -r data && cd impl/rust && cargo clean && cd ../..

# Rust Executable

impl/rust/target/release/rust-impl: data impl/rust/src
	cd impl/rust && cargo build --release

impl/rust/results.csv: impl/rust/target/release/rust-impl
	impl/rust/target/release/rust-impl > impl/rust/results.csv

# Tensorflow

impl/tensorflow/results.csv: impl/tensorflow/src/*.py data
	python3 impl/tensorflow/src/main.py > impl/tensorflow/results.csv
