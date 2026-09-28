"""Tests for the functions in the ``consts`` module."""

import numpy as np
import pandas as pd
import pytest

from pryngles.consts import SPANGLER_COLUMNS, SPANGLER_VEC_GROUPS, Consts


def test_get_physical():
    """``get_physical`` returns the full sorted list of physical constants."""
    assert Consts.get_physical() == [
        "au",
        "aus",
        "cm",
        "d",
        "day",
        "days",
        "deg",
        "g",
        "gram",
        "gyr",
        "hr",
        "jyr",
        "kg",
        "km",
        "kyr",
        "m",
        "massist",
        "mearth",
        "mjupiter",
        "mmars",
        "mmercury",
        "mneptune",
        "mpluto",
        "msaturn",
        "msolar",
        "msun",
        "muranus",
        "mvenus",
        "myr",
        "parsec",
        "pc",
        "ppb",
        "ppm",
        "rad",
        "rearth",
        "rjupiter",
        "rsaturn",
        "rsun",
        "s",
        "solarmass",
        "sunmass",
        "year",
        "years",
        "yr",
        "yrs",
    ]


def test_get_all():
    """``get_all`` returns the full sorted list of numerical constants."""
    assert Consts.get_all() == [
        "BODY_DEFAULTS",
        "BODY_KINDS",
        "DEG",
        "DETECTOR_PROPERTIES",
        "GSI",
        "IN_JUPYTER",
        "LEGACY_PHYSICAL_PROPERTIES",
        "OBSERVER_DEFAULTS",
        "PLANET_DEFAULTS",
        "RAD",
        "REBOUND_CARTESIAN_PROPERTIES",
        "REBOUND_ORBITAL_PROPERTIES",
        "RING_DEFAULTS",
        "ROOTDIR",
        "SAMPLER_CIRCLE_PRESETS",
        "SAMPLER_GEOMETRY_CIRCLE",
        "SAMPLER_GEOMETRY_SPHERE",
        "SAMPLER_MIN_RING",
        "SAMPLER_PRESETS",
        "SAMPLER_SPHERE_PRESETS",
        "SAMPLE_SHAPES",
        "SCATTERERS_CATALOGUE",
        "SCIENCE_LIMB_NORMALIZATIONS",
        "SHADOW_COLOR_LUZ",
        "SHADOW_COLOR_OBS",
        "SPANGLER_AREAS",
        "SPANGLER_COLUMNS",
        "SPANGLER_COLUMNS_DOC",
        "SPANGLER_COL_COPY",
        "SPANGLER_COL_INT",
        "SPANGLER_COL_LUZ",
        "SPANGLER_COL_OBS",
        "SPANGLER_DEBUG_FIELDS",
        "SPANGLER_EPS_BORDER",
        "SPANGLER_EQUIV_COL",
        "SPANGLER_FLUX",
        "SPANGLER_KEY_ORDERING",
        "SPANGLER_KEY_SUMMARY",
        "SPANGLER_LENGTHS",
        "SPANGLER_SOURCE_STATES",
        "SPANGLER_VECTORS",
        "SPANGLER_VEC_GROUPS",
        "SPANGLER_VISIBILITY_STATES",
        "SPANGLES_DARKNESS_COLOR",
        "SPANGLES_SEMITRANSPARENT",
        "SPANGLE_ATMOSPHERIC",
        "SPANGLE_COLORS",
        "SPANGLE_GASEOUS",
        "SPANGLE_GRANULAR",
        "SPANGLE_LIQUID",
        "SPANGLE_SOLID_ICE",
        "SPANGLE_SOLID_ROCK",
        "SPANGLE_STELLAR",
        "STAR_DEFAULTS",
        "T_MODEL_DEFAULTS",
    ]


def test_vec_groups_columns_exist_in_columns():
    """Every column referenced by a vector group exists in SPANGLER_COLUMNS."""
    for group, components in SPANGLER_VEC_GROUPS.items():
        for c in components:
            assert c in SPANGLER_COLUMNS, f"group '{group}' references missing column '{c}'"


def test_vectors_accessor_returns_subframe(spangler_df):
    """``df.vectors.<group>`` returns the (N,3) sub-DataFrame for that group."""
    for group, components in SPANGLER_VEC_GROUPS.items():
        sub = getattr(spangler_df.vectors, group)
        pd.testing.assert_frame_equal(sub, spangler_df[components])


def test_vectors_accessor_to_numpy(spangler_df):
    """``df.vectors.<group>.to_numpy()`` returns the expected (N,3) ndarray."""
    for group, components in SPANGLER_VEC_GROUPS.items():
        arr = getattr(spangler_df.vectors, group).to_numpy()
        np.testing.assert_allclose(arr, spangler_df[components].to_numpy())


def test_vectors_accessor_unknown_group(spangler_df):
    """Accessing an unknown group raises AttributeError listing available groups."""
    with pytest.raises(AttributeError, match="not a known spangler vector group"):
        spangler_df.vectors.not_a_real_group