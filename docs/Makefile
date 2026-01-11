PROJECT = main
BUILD = build

LATEX = pdflatex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

SRC = $(PROJECT).tex sections/abstract.tex sections/intro.tex sections/biblio.tex references.bib

all: pdf

pdf: $(SRC)
	mkdir -p $(BUILD)
	$(LATEX) $(LATEXFLAGS) $(PROJECT).tex
	bibtex $(BUILD)/$(PROJECT)
	$(LATEX) $(LATEXFLAGS) $(PROJECT).tex
	$(LATEX) $(LATEXFLAGS) $(PROJECT).tex

clean:
	rm -rf $(BUILD)

view: pdf
	xdg-open $(BUILD)/$(PROJECT).pdf || open $(BUILD)/$(PROJECT).pdf

.PHONY: all pdf clean view
