"""Wall-clock benchmark of the regression lightcurve workloads, phase by phase.

The systems mirror the fixtures in ``tests/regression/``. Import is timed once
(it is cached after the first import in a process); the system build and
lightcurve are repeated ``--repeat`` times and the median and min of each phase
are reported.

Usage (from the repo root):
    uv run python benchmarks/bench_lightcurve.py --system wasp43b --repeat 3
    uv run python benchmarks/bench_lightcurve.py --system wasp43b --label baseline --out benchmarks/results/results.csv

See ``benchmarks/README.md`` for profiling and comparison workflows.
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

t_import0 = time.perf_counter()
import numpy as np  # noqa: E402
import spiceypy as spy  # noqa: E402

import pryngles as pr  # noqa: E402

t_import = time.perf_counter() - t_import0

PHASES = ["setup", "spangle", "lightcurve", "total"]


# --------------------------------------------------
# Systems (mirroring tests/regression/)
# --------------------------------------------------
def build_quickstart():
    """Star/planet/ring system from tests/regression/test_quickstart_system.py."""
    system = pr.System()
    star = system.add(kind="Star", radius=pr.Consts.rsun / system.ul, limb_coeffs=[0.65])
    planet = system.add(kind="Planet", parent=star, a=0.2, e=0.0, radius=pr.Consts.rsaturn / system.ul)
    system.add(kind="Ring", parent=planet, fi=1.5, fe=2.5, i=30 * pr.Consts.deg)
    system.n_obs = spy.eul2m(np.deg2rad(0.0), np.deg2rad(90.0), 0, 3, 1, 3)[0]
    system.initialize_simulation()

    def lightcurve_kwargs(n_times, polarization):
        period_days = 365.25 * (planet.a**1.5)
        times = np.linspace(0.0, period_days, n_times) * pr.Consts.day / system.ut
        effects = ["polarization"] if polarization else ["reflection"]
        return dict(times=times, effects=effects)

    return system, lightcurve_kwargs


def build_wasp43b():
    """WASP-43b system from tests/regression/test_wasp43b_system.py."""
    period, a_abs, a_rel, inc, ecc, omega = 0.81347753, 0.01526, 4.857, 90, 0.0, 0.0
    detector = {
        "wavelength_min": 1.1e-6,
        "wavelength_max": 1.7e-6,
        "apperture": 0.5,
        "quantum_eff": 0.9,
        "t_cadence": 15 * 60,
        "distance": 100 * pr.Consts.pc,
    }

    R_star_m = a_abs / a_rel * pr.Consts.au
    system = pr.System()
    star = system.add(
        kind="Star",
        m=0.717 * pr.Consts.msun / system.um,
        T_eff=4520,
        radius=R_star_m / system.ul,
        limb_coeffs=[0.65],
    )
    planet = system.add(
        kind="Planet",
        parent=star,
        m=0 * pr.Consts.mjupiter / system.um,
        radius=1.036 * pr.Consts.rjupiter / system.ul,
        a=a_abs,
        inc=pr.consts.DEG * (90 - inc),
        e=ecc,
        omega=pr.consts.DEG * omega,
    )
    system.add(kind="Ring", parent=planet, fi=1.5, fe=2.5, i=pr.consts.DEG * 60, taur=0.4)
    system.n_obs = spy.eul2m(np.deg2rad(omega), np.deg2rad(inc), 0, 3, 1, 3)[0]
    system.initialize_simulation()

    def lightcurve_kwargs(n_times, polarization):
        planet.set_temperature_model(
            {"type": "Zhang-Showman", "params": {"T_night": 1200, "Delta_T": 800, "xi_ratio": 0.3}}
        )
        times = np.linspace(-period / 2, period / 2, n_times) * pr.Consts.day / system.ut
        effects = ["transit", "emission"] + (["polarization"] if polarization else [])
        np.random.seed(42)
        return dict(
            times=times,
            effects=effects,
            signal=detector,
            bandwidth=(detector["wavelength_min"], detector["wavelength_max"]),
        )

    return system, lightcurve_kwargs


SYSTEMS = {"quickstart": build_quickstart, "wasp43b": build_wasp43b}


# --------------------------------------------------
# Timing
# --------------------------------------------------
def run_once(build, n_times, polarization):
    """Build the system and compute the lightcurve; return per-phase seconds."""
    t = {}
    t0 = time.perf_counter()
    system, lightcurve_kwargs = build()
    t1 = time.perf_counter()
    t["setup"] = t1 - t0

    system.spangle_system()
    t2 = time.perf_counter()
    t["spangle"] = t2 - t1

    kwargs = lightcurve_kwargs(n_times, polarization)
    t3 = time.perf_counter()
    system.compute_lightcurve(**kwargs)
    t["lightcurve"] = time.perf_counter() - t3

    t["total"] = t["setup"] + t["spangle"] + t["lightcurve"]
    return t


def git_info():
    """Short commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip() != ""
    except OSError:
        commit, dirty = "", False
    return commit, dirty


def append_csv(path, args, runs):
    """Append one summary row (medians and mins of each phase) to a CSV file."""
    commit, dirty = git_info()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                ["timestamp", "label", "commit", "dirty", "system", "n_times", "polarization", "repeat", "import"]
                + [f"{p}_median" for p in PHASES]
                + [f"{p}_min" for p in PHASES]
            )
        w.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                args.label,
                commit,
                dirty,
                args.system,
                args.n_times,
                not args.no_pol,
                args.repeat,
                f"{t_import:.3f}",
            ]
            + [f"{statistics.median([r[p] for r in runs]):.3f}" for p in PHASES]
            + [f"{min(r[p] for r in runs):.3f}" for p in PHASES]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="wasp43b")
    parser.add_argument("--repeat", type=int, default=3, help="number of build+lightcurve repetitions")
    parser.add_argument("--n-times", type=int, default=11, help="number of lightcurve time steps")
    parser.add_argument("--no-pol", action="store_true", help="skip polarization (quickstart uses reflection instead)")
    parser.add_argument("--label", default="", help="free-text label stored with the results")
    parser.add_argument("--out", default=None, help="append a summary row to this CSV file")
    args = parser.parse_args()

    runs = []
    for i in range(args.repeat):
        r = run_once(SYSTEMS[args.system], args.n_times, not args.no_pol)
        runs.append(r)
        print(f"run {i + 1}/{args.repeat}: " + "  ".join(f"{k} {v:.2f}s" for k, v in r.items()), file=sys.stderr)

    print(f"\nsystem={args.system}  n_times={args.n_times}  polarization={not args.no_pol}  repeat={args.repeat}")
    print(f"import {t_import:.2f}s (first import in this process)")
    print(f"{'phase':<12}{'median':>10}{'min':>10}")
    for p in PHASES:
        vals = [r[p] for r in runs]
        print(f"{p:<12}{statistics.median(vals):>9.2f}s{min(vals):>9.2f}s")

    if args.out:
        append_csv(args.out, args, runs)
        print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
