
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

out/results/python-tf-gpu.csv: out/data impl/python-tf/main.py ./run.sh out/run/python-tf
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || true
	OUT_FILE="out/results/python-tf-gpu.csv" DATA_PATH="out/data" SCRIPT="out/run/python-tf" NAME="Python Tensorflow GPU" bash run.sh

out/results/python-tf-cpu.csv: out/data impl/python-tf/main.py out/run/python-tf ./run.sh
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || true
	OUT_FILE="out/results/python-tf-cpu.csv" DATA_PATH="out/data" SCRIPT="out/run/python-tf" CUDA_VISIBLE_DEVICES='-1' NAME="Python Tensorflow CPU" bash run.sh

out/bin/cuda: impl/cuda/main.cu
	mkdir out 2> /dev/null || true
	mkdir out/bin 2> /dev/null || true
	nvcc impl/cuda/main.cu -o out/bin/cuda -O3 -extra-device-vectorization

out/run/cuda:
	mkdir out 2> /dev/null || true
	mkdir out/run 2> /dev/null || true
	echo "./out/bin/cuda" > out/run/cuda

out/results/cuda.csv: out/bin/cuda out/data out/run/cuda run.sh
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || true
	OUT_FILE="out/results/cuda.csv" DATA_PATH="out/data" SCRIPT="out/run/cuda" NAME="CUDA" bash run.sh

out/run/rust:
	mkdir out 2> /dev/null || true
	mkdir out/run 2> /dev/null || true
	echo "cargo run --release -p rust-impl" > out/run/rust

out/results/rust.csv: out/data $(wildcard impl/rust/src/*.rs) out/run/rust run.sh
	mkdir out 2> /dev/null || true
	mkdir out/results 2> /dev/null || true
	OUT_FILE="out/results/rust.csv" DATA_PATH="out/data" SCRIPT="out/run/rust" NAME="Rust" bash run.sh

out/results/data.csv: out/results/rust.csv out/results/cuda.csv out/results/python-tf-gpu.csv out/results/python-tf-cpu.csv
	echo "epoch,time,variables,bootstraps,model" > out/results/data.csv
	cat out/results/rust.csv >> out/results/data.csv
	cat out/results/cuda.csv >> out/results/data.csv
	cat out/results/python-tf-gpu.csv >> out/results/data.csv
	cat out/results/python-tf-cpu.csv >> out/results/data.csv


##############################################





# Graphs
out/graphs: out/results/data.csv graphs.R
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
