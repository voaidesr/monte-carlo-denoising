PROJECT = main
BUILD = build

LATEX = pdflatex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error -output-directory=$(BUILD)

SRC = $(PROJECT).tex sections/intro.tex

all: pdf

pdf: $(SRC)
	mkdir -p $(BUILD)
	$(LATEX) $(LATEXFLAGS) $(PROJECT).tex
	$(LATEX) $(LATEXFLAGS) $(PROJECT).tex

clean:
	rm -rf $(BUILD)

view: pdf
	xdg-open $(BUILD)/$(PROJECT).pdf || open $(BUILD)/$(PROJECT).pdf

.PHONY: all pdf clean view
