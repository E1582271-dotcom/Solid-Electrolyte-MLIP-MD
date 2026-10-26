PYTHON ?= python
# Run from a checkout without installing: the package is importable from src/.
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: all classify tables figures docs test check lint

all: classify tables figures docs

classify:
	$(PYTHON) -m ssemd classify

tables:
	$(PYTHON) -m ssemd tables

figures:
	$(PYTHON) -m ssemd figures

docs:
	$(PYTHON) -m ssemd docs

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests production

# Rebuild everything from the committed data, then require an unchanged working tree.
check: all test
	$(PYTHON) -m ssemd check-repo
	git diff --exit-code -- data figures docs README.md
