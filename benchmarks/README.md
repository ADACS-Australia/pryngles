# Benchmarks

Tools for timing the pipeline.

`bench_lightcurve.py` runs the same systems as the regression tests in `tests/regression/` (`quickstart` and `wasp43b`), times each phase with `time.perf_counter()`, and reports the median and minimum over several runs.

## Timing

All commands are run from the repo root.

```bash
uv run python benchmarks/bench_lightcurve.py --system wasp43b --repeat 3
```

Output:

```
system=wasp43b  n_times=11  polarization=True  repeat=3
import 4.28s (first import in this process)
phase           median       min
setup            3.71s     3.54s
spangle          1.23s     1.22s
lightcurve      12.70s    12.28s
total           17.66s    17.04s
```

| Phase | What it covers |
|---|---|
| `import` | `import pryngles` (timed once per process) |
| `setup` | creating the `System` and its bodies, including loading the Fourier scattering files, and `initialize_simulation()` |
| `spangle` | `spangle_system()` |
| `lightcurve` | `compute_lightcurve()` |
| `total` | `setup + spangle + lightcurve` (excludes import) |

Options:

| Option | Default | Meaning |
|---|---|---|
| `--system {quickstart,wasp43b}` | `wasp43b` | which regression system to run |
| `--repeat N` | 3 | number of build + lightcurve repetitions |
| `--n-times N` | 11 | number of lightcurve time steps (11 matches the regression tests) |
| `--no-pol` | off | skip polarization (wasp43b: transit + emission only; quickstart: reflection instead) |
| `--label TEXT` | | label stored with the results |
| `--out FILE` | | append a summary row to a CSV file |

## Recording and comparing results

Pass `--out` to append a summary row to a CSV. Each row records the time, label,
git commit, whether the working tree had uncommitted changes, the settings, and
the median and min of every phase:

```bash
uv run python benchmarks/bench_lightcurve.py --system wasp43b --label baseline --out benchmarks/results/results.csv
uv run python benchmarks/bench_lightcurve.py --system wasp43b --no-pol --label baseline --out benchmarks/results/results.csv
```

`benchmarks/results/` is git-ignored, since timings are machine-specific.

When comparing before and after a change:

- use the same `--system`, `--repeat`, `--n-times` and `--no-pol` settings, on the same machine, with other heavy work closed;
- compare medians, and treat differences under about 5% as noise;
- check the change still passes the regression tests: `uv run pytest --regression`.

## Profiling

[pyinstrument](https://pyinstrument.readthedocs.io/) sampling profiler that shows a call tree. `uv run --with` provides it for a single command without adding it to the project:

```bash
# Call tree in the terminal
uv run --with pyinstrument pyinstrument benchmarks/bench_lightcurve.py --repeat 1

# Interactive HTML report
uv run --with pyinstrument pyinstrument -r html -o profile.html benchmarks/bench_lightcurve.py --repeat 1

# Save the run, then render it later with different options
uv run --with pyinstrument pyinstrument -r session -o run.pyisession benchmarks/bench_lightcurve.py --repeat 1
uv run --with pyinstrument pyinstrument --load run.pyisession -r text --hide-regex '.*/(pandas|numpy)/.*'
```

Use `--repeat 1` when profiling.