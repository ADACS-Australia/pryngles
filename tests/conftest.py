"""Shared pytest fixtures and import setup for the pryngles test suite."""

import contextlib
import io
import os
import sys

# The package uses a ``src`` layout. Make ``src`` importable so the test
# suite can run even without an explicit ``pip install`` (the editable
# install also provides it, this is a belt-and-braces fallback).
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Importing the package emits a module-level "Welcome to pryngles vX" banner.
# Capture it once so pytest output stays clean.
with contextlib.redirect_stdout(io.StringIO()):
    import pryngles  # noqa: F401
