# Regression Tests

This directory contains regression tests for the pryngles codebase. They
capture the numerical output of the lightcurve pipeline (spangler state,
lightcurve data, system metadata, detector signal) and compare it against
stored "golden" files using [pytest-regressions](https://github.com/ESSS/pytest-regressions).

The regression tests are **skipped by default** because they are slow.
Run them explicitly with the `--regression` flag.

## Running the tests

All commands below use `uv` (the project's package manager). Install the
dev dependencies first if you haven't already:

```bash
uv sync --group dev
```

### Run the regression suite

```bash
uv run pytest --regression
```

### Run a single regression file

```bash
uv run pytest --regression tests/regression/test_quickstart_system.py
uv run pytest --regression tests/regression/test_wasp43b_system.py
```

### Show live progress

The system fixtures compute lightcurves, which can take a while (the
wasp43b suite takes ~70s). pytest captures output by default, so the
tqdm progress bars are hidden until a test finishes. Pass `-s` (or
`--capture=no`) to see the live progress and confirm tests are still
running:

```bash
uv run pytest --regression -s
```

## Golden files

Each test file has a sibling directory holding its golden files, e.g.
`tests/regression/test_quickstart_system/`. These are generated on the
first run and compared on subsequent runs.

- **First run**: pytest-regressions reports a failure because the golden
  file doesn't exist yet. Re-run the same command to pass.
- **Regenerate**: pass `--force-regen` to overwrite the golden files
  with the current output (use when the expected output has legitimately
  changed):

```bash
uv run pytest --regression --force-regen
```

## Ignored warnings

`DeprecationWarning` and `RuntimeWarning` are ignored globally (see
`filterwarnings` in `pyproject.toml`) to keep the test output clean —
the lightcurve computation emits numpy warnings that are expected and
unrelated to the regression checks.

## Layout

- `utils.py` — capture functions that turn system/lightcurve objects
  into plain data suitable for pytest-regressions.
- `test_quickstart_system.py` — regression test for the quickstart
  pipeline (Star/Planet/Ring, polarization effect).
- `test_wasp43b_system.py` — more involved regression test (WASP-43b
  system with temperature model and detector).
