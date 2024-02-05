target/release/rust-impl: Cargo.toml Cargo.lock src/*
	cargo build --release

results.csv: target/release/rust-impl
	./target/release/rust-impl > results.csv

clean:
	cargo clean && rm results.csv
