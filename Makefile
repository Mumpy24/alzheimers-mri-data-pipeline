.PHONY: install test lint demo figure clean

install:
	pip install -e ".[dev,viz]"

test:
	pytest --cov

lint:
	ruff check .
	ruff format --check .
	mypy src

demo:
	admri synth data/bids --inject-faults --overwrite
	admri -v run data/bids data/out --workers 4 --config configs/default.toml
	admri audit data/out

figure:
	python scripts/make_figure.py

clean:
	rm -rf data .pytest_cache .mypy_cache .ruff_cache .coverage build *.egg-info src/*.egg-info
