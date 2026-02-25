.PHONY: install run debug clean lint lint-strict

install:
	uv sync
	uv pip install -e

run:
	uv run python -m src.main

debug:
	uv run python -m pdb src/main.python

clean:
	rm -rf __pychache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
	uv run flake8 scr
	uv run mypy --strict src

index:
	uv run python -m src index --chunk_size 2000