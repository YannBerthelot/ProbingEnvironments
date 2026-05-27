SHELL=/bin/bash

# Strict CPU isolation -- prevents tests from touching GPU when a live
# experiment is running on this machine. CUDA_VISIBLE_DEVICES="" hides the
# GPU from the CUDA driver entirely so CUDA init can't probe it.
CPU_ENV := CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu

.PHONY: ci ci-tox ci-linting ci-coverage test help

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Canonical local-CI -- runs tox which covers linting + tests + coverage,
# matching what .github/workflows/poetry_build.yaml runs in CI.
# Coverage gate per tox.ini ([testenv:coverage]) is --fail-under 75.
# ---------------------------------------------------------------------------

ci: ci-tox  ## Full local CI (matches .github/workflows/poetry_build.yaml -> tox)

ci-tox:  ## Run the full tox envlist (py312, linting, coverage)
	$(CPU_ENV) poetry run tox

ci-linting:  ## Only the linting (pre-commit) env
	poetry run tox -e linting

ci-coverage:  ## Only the coverage env (--fail-under 75)
	$(CPU_ENV) poetry run tox -e coverage

# ---------------------------------------------------------------------------
# Convenience: fast CPU-only pytest run without the coverage gate.
# ---------------------------------------------------------------------------

test:  ## Run pytest on CPU only (no coverage gate)
	$(CPU_ENV) poetry run pytest --tb=short --disable-warnings
