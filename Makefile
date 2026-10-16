.PHONY: lint clean

PYTHON ?= python3

VENV_LINT = .venv-lint

$(VENV_LINT):
	$(PYTHON) -m venv $(VENV_LINT)
# 	TODO: make .[lint]
	. $(VENV_LINT)/bin/activate && pip install .[ci]

ruff-format:
	. $(VENV_LINT)/bin/activate && ruff format --check .

ruff-check:
	. $(VENV_LINT)/bin/activate && ruff check .

ruff-fix:
	. $(VENV_LINT)/bin/activate && ruff check . --fix

ruff-fix-unsafe:
	. $(VENV_LINT)/bin/activate && ruff check . --fix --unsafe-fixes

mypy:
	. $(VENV_LINT)/bin/activate && mypy -p ufl
	. $(VENV_LINT)/bin/activate && mypy test/
	. $(VENV_LINT)/bin/activate && MYPYPATH=test mypy demo/

lint: $(VENV_LINT) ruff-format ruff-check mypy

clean:
	rm -rf $(VENV_LINT) .mypy_cache .pytest_cache .ruff_cache
