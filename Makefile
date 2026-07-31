PYTHON := uv run python
SRC := src

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m $(SRC)

debug:
	$(PYTHON) -m pdb -m $(SRC)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf data/output

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
