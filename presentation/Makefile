LATEXMK ?= latexmk
ENGINE ?= -lualatex
TEXMFVAR := $(CURDIR)/build/texmf-var
OSFONTDIR ?= $(HOME)/Library/Fonts:/Library/Fonts:/System/Library/Fonts

.PHONY: all clean

all: nma-template-demo.pdf internal-states-gating.pdf

nma-template-demo.pdf: demo.tex beamerthemeNMA.sty
	mkdir -p build $(TEXMFVAR)
	TEXMFVAR=$(TEXMFVAR) OSFONTDIR="$(OSFONTDIR)" $(LATEXMK) $(ENGINE) -interaction=nonstopmode -halt-on-error -outdir=build demo.tex
	cp build/demo.pdf $@

internal-states-gating.pdf: internal-states-gating.tex beamerthemeNMA.sty
	mkdir -p build $(TEXMFVAR)
	TEXMFVAR=$(TEXMFVAR) OSFONTDIR="$(OSFONTDIR)" $(LATEXMK) $(ENGINE) -interaction=nonstopmode -halt-on-error -outdir=build internal-states-gating.tex
	cp build/internal-states-gating.pdf $@

clean:
	$(LATEXMK) -C -outdir=build demo.tex
	$(LATEXMK) -C -outdir=build internal-states-gating.tex
	rm -rf build
