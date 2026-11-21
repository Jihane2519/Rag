export HF_HOME=/goinfre/jbounou/.cache/huggingface
export UV_CACHE_DIR=/goinfre/$USER/.cache/uv

ARG ?= Haha

install:
	uv sync

run:
	uv run python -m src $(ARG)

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

lint:
	flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 src/
	mypy src/ --strict
