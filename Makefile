.PHONY: venv lint clean

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_STAMP := $(VENV)/.installed

venv: $(VENV_STAMP)

$(VENV_STAMP):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install ".[ci]"
	touch $(VENV_STAMP)

lint: venv
	$(VENV_PYTHON) -m ruff check .
	$(VENV_PYTHON) -m ruff format --check .
	$(VENV_PYTHON) -m mypy -p ufl
	$(VENV_PYTHON) -m mypy test/
	MYPYPATH=test $(VENV_PYTHON) -m mypy demo/

clean:
	rm -rf $(VENV) .mypy_cache .pytest_cache .ruff_cache build dist *.egg-info
