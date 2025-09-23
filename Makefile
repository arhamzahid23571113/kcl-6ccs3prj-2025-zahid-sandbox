.PHONY: pdf open clean

pdf:
	@cd report && latexmk -pdf -halt-on-error main.tex && ln -sf build/main.pdf main.pdf

open:
	@open report/main.pdf

clean:
	@cd report && latexmk -C && rm -rf build/*
