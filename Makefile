.PHONY: lint

lint:
	ruff check .
	ruff format --check .
	mypy -p ufl
	mypy test/
	MYPYPATH=test mypy demo/
