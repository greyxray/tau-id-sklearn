.PHONY: test

test: test/*.py
	python -m unittest test.test_data