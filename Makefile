SHELL=/bin/bash

# Strict CPU isolation -- prevents tests from touching GPU when a live
# experiment is running on this machine. CUDA_VISIBLE_DEVICES="" hides the
# GPU from the CUDA driver entirely so CUDA init can't probe it.
CPU_ENV := CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu

.PHONY: ci lint test test-all docs build help

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Canonical local CI -- the same steps as .github/workflows/ci.yml.
# Coverage gate: fail_under in [tool.coverage.report] (pyproject.toml).
# ---------------------------------------------------------------------------

ci: lint test-all docs  ## Full local CI (matches .github/workflows/ci.yml)

lint:  ## Linters, as pre-commit runs them
	uv run pre-commit run --all-files

test:  ## Small tier only, on CPU (the inner loop)
	$(CPU_ENV) uv run pytest --tb=short --disable-warnings

test-all:  ## Small and medium tiers with the coverage gate, on CPU
	$(CPU_ENV) uv run coverage run -m pytest -q -m "not large"
	uv run coverage report

docs:  ## Build the documentation site strictly
	uv run --group docs mkdocs build --strict

build:  ## Build the wheel and the sdist into dist/
	uv build --no-sources
