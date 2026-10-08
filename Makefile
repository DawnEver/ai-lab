PY ?= .venv/Scripts/python.exe

.PHONY: install-dev lint fmt-check test live verify
install-dev:
	uv venv .venv && uv pip install --python $(PY) -e ".[dev]"
lint:
	$(PY) -m ruff check .
fmt-check:
	$(PY) -m ruff format --check .
test:
	$(PY) -m pytest -q
live:
	$(PY) -m pytest -q -m live
verify: lint fmt-check test
