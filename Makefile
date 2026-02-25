.PHONY: install run debug clean lint lint-strict

PYTHON = uv run python
SRC = src

install:
	uv sync
	uv pip install -e .

run:
	$(PYTHON) -m $(SRC).main

serve:
	ollama serve &

debug:
	$(PYTHON) -m pdb $(SRC)/main.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .uv_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
	uv run flake8 src
	uv run mypy --strict src

index:
	uv run python -m src index

setup:
	ollama serve & sleep 5 && ollama pull qwen3:0.6b

moulinette:
	uv run python -m moulinette evaluate_student_search_results \
		--student_answer_path data/output/search_results/dataset_docs_public.json \
		--dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json 

answer:
	uv run python -m src answer_dataset \
		--student_search_results_path data/output/search_results/dataset_docs_public.json \
		--save_directory data/output/search_results_and_answer
