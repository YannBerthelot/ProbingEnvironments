"""Shared test setup."""

import os

# Probes are tiny, so the suite runs on CPU (TESTING.md A.9). Set before any
# test module imports JAX.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
