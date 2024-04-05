
# Paper
thesis/document.pdf: thesis/document.toc thesis/document.tex
	cd thesis && pdflatex -halt-on-error document.tex >> /dev/null

thesis/document.aux: thesis/document.tex
	cd thesis && pdflatex -halt-on-error document.tex >> /dev/null

thesis/document.blg: thesis/document.aux thesis/refs.bib
	cd thesis && bibtex document.aux >> /dev/null

thesis/document.toc: thesis/document.blg
	cd thesis && pdflatex -halt-on-error document.tex >> /dev/null

paper/clean:
	cd paper && rm document.aux document.bbl document.blg document.log document.pdf document.toc


##############################################
# NEW CONFIGURATIONS

out/data: data.toml data.py
	mkdir out 2> /dev/null || true
	rm -r out/data || true
	mkdir out/data
	python3 data.py

out/run/python-tf:
	mkdir out 2> /dev/null || true
	mkdir out/run 2> /dev/null || true
	echo "python3 impl/python-tf/main.py" > out/run/python-tf

out/results/python-tf-gpu: out/data impl/python-tf/main.py ./run.sh out/run/python-tf
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || rm -r out/results/python-tf-gpu 2> /dev/null || true
	mkdir out/results/python-tf-gpu
	OUT_PATH="out/results/python-tf-gpu" DATA_PATH="out/data" SCRIPT="out/run/python-tf" NAME="Python Tesnorflow GPU" bash run.sh

out/results/python-tf-cpu: out/data impl/python-tf/main.py out/run/python-tf ./run.sh
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || rm -r out/results/python-tf-cpu 2> /dev/null || true
	mkdir out/results/python-tf-cpu
	OUT_PATH="out/results/python-tf-cpu" DATA_PATH="out/data" SCRIPT="out/run/python-tf" CUDA_VISIBLE_DEVICES='-1' NAME="Python Tensorflow CPU" bash run.sh

out/bin/cuda: impl/cuda/main.cu
	mkdir out 2> /dev/null || true
	mkdir out/bin 2> /dev/null || true
	nvcc impl/cuda/main.cu -o out/bin/cuda -O3 -extra-device-vectorization

out/run/cuda:
	mkdir out 2> /dev/null || true
	mkdir out/run 2> /dev/null || true
	echo "./out/bin/cuda" > out/run/cuda

out/results/cuda: out/bin/cuda out/data out/run/cuda run.sh
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || rm -r out/results/cuda 2> /dev/null || true
	mkdir out/results/cuda
	OUT_PATH="out/results/cuda" DATA_PATH="out/data" SCRIPT="out/run/cuda" NAME="Cuda" bash run.sh

out/run/rust:
	mkdir out 2> /dev/null || true
	mkdir out/run 2> /dev/null || true
	echo "cargo run --release -p rust-impl" > out/run/rust

out/results/rust: out/data $(wildcard impl/rust/src/*.rs) out/run/rust run.sh
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || rm -r out/results/rust 2> /dev/null || true
	mkdir out/results/rust
	OUT_PATH="out/results/rust" DATA_PATH="out/data" SCRIPT="out/run/rust" NAME="Rust" bash run.sh

##############################################





# Graphs
out/graphs: out/results/cuda out/results/python-tf-cpu out/results/python-tf-gpu out/results/rust graphs.R
	rm -r out/graphs || true
	mkdir out/graphs
	R < graphs.R --no-save


# Misc
data: data.toml data.py
	python3 data.py

rmdata:
	rm -r data 2> /dev/null || true

clean:
	rm -r bin 2> /dev/null || true
	rm -r data 2> /dev/null || true
	rm -r out 2> /dev/null || true
	cargo clean
