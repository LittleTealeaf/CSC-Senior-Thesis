#!/bin/bash

./build.sh
gio open ./document.pdf

while inotifywait -e move_self document.tex; do
	./build.sh
done
