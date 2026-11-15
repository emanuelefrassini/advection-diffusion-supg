.PHONY: test plots
test:
	python -m pytest -q
plots:
	python scripts/run_comparison.py
	python scripts/run_refinement.py
