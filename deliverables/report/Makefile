# Makefile for compiling a LaTeX document using the acmart template

# The name of the LaTeX source file (without extension)
TEXFILE = report

# Compiler and tools
LATEX = pdflatex
BIBTEX = bibtex
BIBLATEX = biber

# ACM-related files
ACMCLASS = acmart.cls
BIBFILE = acmart.bib
BSTFILE = ACM-Reference-Format.bst

# PDF file to be generated
PDF = $(TEXFILE).pdf

# Targets to generate PDF
all: $(PDF)

# Rule to generate PDF from .tex
$(TEXFILE).pdf: $(TEXFILE).tex $(ACMCLASS) $(BSTFILE) $(BIBFILE)
	$(LATEX) $(TEXFILE).tex        # First LaTeX run
	$(BIBTEX) $(TEXFILE)           # Run BibTeX (for managing references)
	$(LATEX) $(TEXFILE).tex        # Second LaTeX run (to resolve references)
	$(LATEX) $(TEXFILE).tex        # Third LaTeX run

# Clean up auxiliary files generated during compilation
clean:
	rm -f $(TEXFILE).aux $(TEXFILE).log $(TEXFILE).out $(TEXFILE).bbl $(TEXFILE).blg $(TEXFILE).toc

# Remove all generated files, including the PDF
distclean: clean
	rm -f $(TEXFILE).pdf

# Clean all auxiliary and sample files
docclean:
	$(RM)  *.log *.aux \
	*.cfg *.glo *.idx *.toc \
	*.ilg *.ind *.out *.lof \
	*.lot *.bbl *.blg *.gls *.cut *.hd \
	samples/$(ACMCLASS) \
	samples/ACM-Reference-Format.bst \
	samples/*.log samples/*.aux samples/*.out \
	samples/*.bbl samples/*.blg samples/*.cut \
	samples/acm-jdslogo.png \
	samples/*.run.xml samples/*.bcf $(SAMPLEBIBLATEXFILES)

.PHONY: all clean distclean docclean

