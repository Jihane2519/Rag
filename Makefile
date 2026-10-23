install:
		uv sync

run:
		uv run python -m src

debug:
		uv run python -m pdb -m src

clean:
		find . -type d -name "__pycache__" -exec rm -rf {} +
		find . -type d -name ".mypy_cache" -exec rm -rf {} +


lint-strict:
		mypy src/ --strict
		flake8 src/

lint:
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	flake8 src/