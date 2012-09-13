INTERPRETER=pdflatex
SOURCE=sphere-doc.tex
PDF=$(SOURCE:.tex=.pdf)

$(PDF):	$(SOURCE)
	$(INTERPRETER) $< $@

clean:
	$(RM) *.{out,toc,log,aux}
