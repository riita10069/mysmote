# command
LATEX = platex
DVIPDF = dvipdfmx

# files
TARGET = main
SRC = $(TARGET).tex
DVI = $(TARGET).dvi
PDF = $(TARGET).pdf

build:
	$(LATEX) $(SRC)
	$(DVIPDF) $(DVI)
clean:
	rm $(WORK_DIR)/*.log
rm:
	rm $(WORK_DIR)/main.pdf
