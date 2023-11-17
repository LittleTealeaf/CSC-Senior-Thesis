#!/bin/bash
pdflatex document.tex && bibtex document.aux && pdflatex document.tex && pdflatex document.tex
