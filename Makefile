.PHONY: test

test: test/*.py
	python -m unittest discover