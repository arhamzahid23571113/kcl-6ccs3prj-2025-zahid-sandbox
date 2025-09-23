.PHONY: pdf clean

pdf:
	@cd report && latexmk -pdf -halt-on-error main.tex

clean:
	@cd report && latexmk -C && rm -rf build/*
