#!/usr/bin/env python
"""
Regression utilities for the pryngles test suite.

These helpers capture the deterministic numerical output of a pryngles
pipeline (lightcurve + final spangler state + system metadata) and compare
it against a stored "golden" reference, so that optimisations can be
verified not to change the results.

The pipeline is deterministic (verified: two runs produce byte-identical
output). The pytest integration uses pytest-regressions, which compares
with a configurable tolerance.

Design goals
------------
* Schema-independent capture: the golden output is identical whether the code
  stores a vector as a single object column ("center_ecl") or as three float
  component columns ("center_ecl_x/y/z"). This lets us refactor the
  DataFrame layout (e.g. splitting object columns for vectorisation) without
  invalidating the golden reference.
* Robust to column renames / reordering: columns are matched by a canonical
  name (see ``SPANGLER_SCHEMA``), and the comparison walks the captured
  structure by path, so moved values are still compared.
* Conversion to plain data happens at capture time (``_to_plain``), so the
  golden file is stable across code changes: adding/removing attributes on
  internal classes does not alter the captured output as long as the
  numerical values are unchanged.

The captured output includes:
    - system metadata (bodies, n_obs, units, conversion factors)
    - the full lightcurve dict (times, total_flux, per-effect DataFrames)
    - the final Spangler `data` DataFrame (all columns), so intermediate
      geometry/state changes are caught even if they cancel out in the flux.
"""
import numpy as np
import pandas as pd
import spiceypy as spy

import pryngles as pr

# ---------------------------------------------------------------------------
# Per-DataFrame capture schema
# ---------------------------------------------------------------------------
# Each DataFrame being captured can have its own schema describing how its
# columns map to canonical names. This keeps the generic ``_capture_dataframe``
# free of any one DataFrame's layout, so the lightcurve DataFrames (which have
# no vector columns) are captured generically while the spangler DataFrame
# gets its vector handling.
#
# A schema is a dict with optional keys:
#   vector_groups: {canonical_base: [component_cols, ...]}
#       Expand a vector stored as a single object column of 3-vectors into
#       its component columns (``base_x/y/z``) so the golden output is
#       numeric and comparable by ``num_regression``. If the code already
#       stores the vector as separate component columns, those are captured
#       directly. Either way the golden output uses the component names,
#       so the capture is independent of how the code lays out the vector.
#   column_aliases: {canonical_name: [alt_names, ...]}
#       Keep comparing a value even if a column is renamed during
#       refactoring. The first name present in the data wins.
#   excluded_columns: set of column names to skip.

# Schema for the Spangler data DataFrame (the only one with vector columns).
SPANGLER_SCHEMA = {
    "vector_groups": {
        "center_equ": ["center_equ_x", "center_equ_y", "center_equ_z"],
        "center_ecl": ["center_ecl_x", "center_ecl_y", "center_ecl_z"],
        "center_obs": ["center_obs_x", "center_obs_y", "center_obs_z"],
        "center_int": ["center_int_x", "center_int_y", "center_int_z"],
        "center_luz": ["center_luz_x", "center_luz_y", "center_luz_z"],
        "ns_equ": ["ns_equ_x", "ns_equ_y", "ns_equ_z"],
        "ns_ecl": ["ns_ecl_x", "ns_ecl_y", "ns_ecl_z"],
        "ns_obs": ["ns_obs_x", "ns_obs_y", "ns_obs_z"],
        "ns_int": ["ns_int_x", "ns_int_y", "ns_int_z"],
        "ns_luz": ["ns_luz_x", "ns_luz_y", "ns_luz_z"],
        "wx_ecl": ["wx_ecl_x", "wx_ecl_y", "wx_ecl_z"],
        "wy_ecl": ["wy_ecl_x", "wy_ecl_y", "wy_ecl_z"],
        "n_int": ["n_int_x", "n_int_y", "n_int_z"],
        "n_int_ecl": ["n_int_ecl_x", "n_int_ecl_y", "n_int_ecl_z"],
        "n_obs": ["n_obs_x", "n_obs_y", "n_obs_z"],
        "n_luz": ["n_luz_x", "n_luz_y", "n_luz_z"],
    },
    "column_aliases": {
        # example: "albedo_gray_normal": ["albedo_gray_normal", "albedo_normal"],
    },
    "excluded_columns": set(),
}


# ---------------------------------------------------------------------------
# Plain-data conversion (applied at capture time)
# ---------------------------------------------------------------------------
# The capture functions below produce plain Python data (lists, dicts,
# scalars, strings) so the output is deterministic and YAML-serializable.
# ``_to_plain`` converts the few non-plain values that can appear in the
# captured output -- numpy arrays/scalars and generic objects (e.g. the
# scatterer surface objects in the spangler data) -- into plain data.
# Generic objects are reduced to their class name, which is the only
# deterministic information worth comparing for them.
def _to_plain(value):
    """Convert a value to plain Python data for regression comparison.

    numpy arrays and scalars become plain lists / Python scalars, and
    generic objects (e.g. scatterer surfaces) become a dict holding only
    their class name. This keeps the captured output deterministic and
    YAML-serializable without a separate normalization pass.
    """
    if isinstance(value, np.ndarray):
        return [_to_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if hasattr(value, '__dict__'):
        return {'_class': type(value).__name__}
    return str(value)


def _capture_dataframe(df, schema=None):
    """Capture a DataFrame in a schema-independent way.

    ``schema`` is a per-DataFrame dict (see the "Per-DataFrame capture
    schema" section) describing how this particular DataFrame's columns map
    to canonical names. If omitted, the DataFrame is captured generically
    (all columns as plain lists).

    Vector quantities stored as a single object column of 3-vectors are
    expanded into their component columns (``base_x/y/z``) so the golden
    output is numeric and comparable by ``num_regression``. If the code
    already stores them as separate component columns, those are captured
    directly. Either way the golden output uses the component names, so
    the capture is independent of how the code lays out the vectors.
    Renamed columns are matched back to their canonical name via
    ``column_aliases``. All other columns are captured as plain lists.

    MultiIndex columns (e.g. the lightcurve effect DataFrames, whose
    columns are ``(body, effect)`` tuples) are flattened into a single
    readable name joined by ``/`` (e.g. ``Planet/polarization``).
    """
    schema = schema or {}
    vector_groups = schema.get('vector_groups', {})
    column_aliases = schema.get('column_aliases', {})
    excluded = schema.get('excluded_columns', set())
    state = {}

    # Map every actual column to its canonical name (identity if no alias).
    canonical_of = {}
    for col in df.columns:
        canonical_of[col] = col
    for canonical, aliases in column_aliases.items():
        for alias in aliases:
            if alias in df.columns:
                canonical_of[alias] = canonical

    for col in df.columns:
        if col in excluded:
            continue
        # A canonical vector stored as a single object column of 3-vectors
        # is expanded into its component columns so the golden output is
        # numeric (and identical whether the code stores the vector as one
        # object column or as three float components).
        if col in vector_groups:
            comps = vector_groups[col]
            rows = df[col].tolist()
            for i, comp in enumerate(comps):
                state[comp] = [_to_plain(row[i]) for row in rows]
            continue
        # Flatten MultiIndex columns into a readable single name.
        if isinstance(col, tuple):
            name = "-".join(str(c) for c in col)
        else:
            name = canonical_of[col]
        state[name] = [_to_plain(v) for v in df[col].tolist()]
    return state


def capture_spangler_state(sg):
    """Capture the full Spangler data DataFrame in a schema-independent way."""
    return _capture_dataframe(sg.data, SPANGLER_SCHEMA)


def capture_lightcurve(lightcurve):
    """Capture a lightcurve dict, treating its DataFrames uniformly.

    The lightcurve dict contains per-effect DataFrames keyed by effect
    name (e.g. ``polarization``) plus a ``scattering`` DataFrame that is
    populated whenever polarization is requested. ``lightcurve['effects']``
    only lists the *requested* effects, so we capture every DataFrame
    value in the dict (which includes ``scattering``) rather than only
    the requested ones, ensuring no effect output is missed.

    If a detector signal was simulated (``lightcurve['signal']``), it is
    captured too. The signal is a dict of arrays (``times``,
    ``signal_flux``, ``signal_error``) and is stochastic, so the caller
    must seed the RNG before computing the lightcurve for a reproducible
    golden file.
    """
    result = {
        'times': _to_plain(lightcurve['times']),
        'total_flux': _to_plain(lightcurve['total_flux']),
        'effects': _to_plain(lightcurve['effects']),
        'bodies': _to_plain(lightcurve["bodies"]),
        'observer': {
            'n_obs': _to_plain(lightcurve["observer"]["n_obs"]),
            'direction': _to_plain(lightcurve["observer"]["direction"]),
        },
        'bandwidth': _to_plain(lightcurve['bandwidth']),
    }
    # Capture every DataFrame-valued key (requested effects plus the
    # always-present ``scattering``), so no effect output is missed.
    result.update({
        key: _capture_dataframe(value)
        for key, value in lightcurve.items()
        if isinstance(value, pd.DataFrame)
    })
    # Capture the detector signal (a dict of arrays) if present.
    if 'signal' in lightcurve:
        result['signal'] = {
            key: _to_plain(value)
            for key, value in lightcurve['signal'].items()
        }
    return result


def capture_detector_signal(system):
    """Capture the detector's configuration, deterministic properties and signal.

    The detector's configuration (wavelength range, aperture, quantum
    efficiency, cadence, distance) and its ``normal_flux`` (computed from
    the source star and detector geometry) are deterministic. The signal
    arrays (``times``, ``signal_flux``, ``signal_error``) are stochastic,
    so the caller must seed the RNG before generating them for a
    reproducible golden file.
    """
    detector = system.detector
    return {
        'wavelength_min': _to_plain(detector.wavelength_min),
        'wavelength_max': _to_plain(detector.wavelength_max),
        'apperture': _to_plain(detector.apperture),
        'quantum_eff': _to_plain(detector.quantum_eff),
        't_cadence': _to_plain(detector.t_cadence),
        'distance': _to_plain(detector.distance),
        'normal_flux': _to_plain(detector.normal_flux),
        'times': _to_plain(detector.times),
        'signal_flux': _to_plain(detector.signal_flux),
        'signal_error': _to_plain(detector.signal_error),
    }


def capture_system_metadata(system):
    """Capture deterministic system-level metadata."""
    return {
        'n_obs': _to_plain(system.n_obs),
        'ul': _to_plain(system.ul),
        'um': _to_plain(system.um),
        'ut': _to_plain(system.ut),
        'observer': {
            'd_obs': _to_plain(getattr(system, 'd_obs', None)),
            'd_luz': _to_plain(getattr(system, 'd_luz', None)),
            'rqf_obs': _to_plain(getattr(system, 'rqf_obs', None)),
            'rqf_luz': _to_plain(getattr(system, 'rqf_luz', None)),
            'alpha_obs': _to_plain(getattr(system, 'alpha_obs', None)),
            'center_obs': _to_plain(getattr(system, 'center_obs', None)),
        },
        'bodies': {
            name: {
                'kind': body.kind,
                'radius': _to_plain(body.radius),
                'a': _to_plain(getattr(body, 'a', None)),
                'e': _to_plain(getattr(body, 'e', None)),
                'fi': _to_plain(getattr(body, 'fi', None)),
                'fe': _to_plain(getattr(body, 'fe', None)),
                'i': _to_plain(getattr(body, 'i', None)),
            }
            for name, body in system.bodies.items()
        },
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _walk(d, prefix=''):
    """Yield (path, value) leaf pairs, keeping lists atomic.

    Unlike a full flatten, lists (which represent a column of values
    across rows) are yielded as a single leaf rather than exploded into
    per-element paths. This lets ``num_regression`` receive each column
    as one 1D array.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}/{k}")
    else:
        yield prefix, d


def _to_1d_array(value):
    """Return ``value`` as a 1D float array, or ``None`` if not numeric.

    Numeric scalars become a single-element array; numeric lists (and
    lists of vectors) are flattened to 1D. Non-numeric values (strings,
    booleans, scatterer class dicts) return ``None`` so they can be
    routed to ``data_regression`` instead.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.asarray([value], dtype=float)
    if isinstance(value, (list, tuple)):
        flat = []
        for v in value:
            if isinstance(v, (list, tuple)):
                flat.extend(v)
            else:
                flat.append(v)
        if flat and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in flat
        ):
            return np.asarray(flat, dtype=float)
    return None


# ---------------------------------------------------------------------------
# pytest integration (pytest-regressions)
# ---------------------------------------------------------------------------
# The capture functions above produce plain Python data and are callable
# from any script. For use with pytest, each test captures one part of the
# output and checks it with pytest-regressions' ``num_regression``
# (numerical data) and ``data_regression`` (metadata) fixtures. This gives
# you:
#   * automatic golden-file generation on first run,
#   * rich diff reporting on mismatch,
#   * integration with the existing pytest test suite.
#
# ``_split_capture`` splits a captured dict into its numerical columns
# (for ``num_regression``, as 1D arrays) and everything else (for
# ``data_regression``).
#
# The golden files are stored next to the test (in the test's data
# directory) and are regenerated with ``pytest --force-regen``.
def _split_capture(captured):
    """Split a captured dict into (numerical, metadata) parts.

    ``num_regression`` expects a dict of name -> 1D array of numbers, so
    each numeric column (a list of values across rows) is converted to a
    1D array. Everything else (strings, booleans, scatterer class names,
    nested structure) goes to ``data_regression``.
    """
    numerical = {}
    metadata = {}
    for path, value in _walk(captured):
        arr = _to_1d_array(value)
        if arr is not None:
            numerical[path] = arr
        else:
            metadata[path] = value
    return numerical, metadata


