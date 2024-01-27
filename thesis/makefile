document.pdf: document.toc document.tex
	pdflatex document.tex >> /dev/null

document.aux: document.tex
	pdflatex document.tex >> /dev/null

document.blg: document.aux refs.bib
	bibtex document.aux >> /dev/null

document.toc: document.blg
	pdflatex document.tex >> /dev/null

clean:
	rm document.aux document.bbl document.blg document.log document.pdf document.toc
