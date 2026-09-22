"""Tests for the functions in the ``consts`` module."""

import pytest

import pryngles as pr


def test_get_physical():
    """``get_physical`` returns the full sorted list of physical constants."""
    assert pr.Consts.get_physical() == [
        "au", "aus", "cm", "d", "day", "days", "deg", "g", "gram", "gyr",
        "hr", "jyr", "kg", "km", "kyr", "m", "massist", "mearth", "mjupiter",
        "mmars", "mmercury", "mneptune", "mpluto", "msaturn", "msolar", "msun",
        "muranus", "mvenus", "myr", "parsec", "pc", "ppb", "ppm", "rad",
        "rearth", "rjupiter", "rsaturn", "rsun", "s", "solarmass", "sunmass",
        "year", "years", "yr", "yrs",
    ]


def test_get_all():
    """``get_all`` returns the full sorted list of numerical constants."""
    assert pr.Consts.get_all() == [
        "ABC", "BODY_DEFAULTS", "BODY_KINDS", "DATA_INDEX", "DEG",
        "DETECTOR_PROPERTIES", "DOUBLE", "FILE", "GSI", "HASH_MAXSIZE",
        "HTML", "IN_JUPYTER", "LEGACY_PHYSICAL_PROPERTIES", "NORMFACTOR",
        "OBSERVER_DEFAULTS", "PDOUBLE", "PLANET_DEFAULTS", "PPDOUBLE",
        "PPPDOUBLE", "RAD", "REBOUND_CARTESIAN_PROPERTIES",
        "REBOUND_ORBITAL_PROPERTIES", "RING_DEFAULTS", "ROOTDIR",
        "SAMPLER_CIRCLE_PRESETS", "SAMPLER_GEOMETRY_CIRCLE",
        "SAMPLER_GEOMETRY_SPHERE", "SAMPLER_MIN_RING", "SAMPLER_PRESETS",
        "SAMPLER_SPHERE_PRESETS", "SAMPLE_SHAPES", "SCATTERERS_CATALOGUE",
        "SCIENCE_LIMB_NORMALIZATIONS", "SHADOW_COLOR_LUZ", "SHADOW_COLOR_OBS",
        "SPANGLER_AREAS", "SPANGLER_COLUMNS", "SPANGLER_COLUMNS_DOC",
        "SPANGLER_COL_COPY", "SPANGLER_COL_INT", "SPANGLER_COL_LUZ",
        "SPANGLER_COL_OBS", "SPANGLER_DEBUG_FIELDS", "SPANGLER_EPS_BORDER",
        "SPANGLER_EQUIV_COL", "SPANGLER_FLUX", "SPANGLER_KEY_ORDERING",
        "SPANGLER_KEY_SUMMARY", "SPANGLER_LENGTHS", "SPANGLER_SOURCE_STATES",
        "SPANGLER_VECTORS", "SPANGLER_VISIBILITY_STATES",
        "SPANGLES_DARKNESS_COLOR", "SPANGLES_SEMITRANSPARENT",
        "SPANGLE_ATMOSPHERIC", "SPANGLE_COLORS", "SPANGLE_GASEOUS",
        "SPANGLE_GRANULAR", "SPANGLE_LIQUID", "SPANGLE_SOLID_ICE",
        "SPANGLE_SOLID_ROCK", "SPANGLE_STELLAR", "STAR_DEFAULTS",
        "T_MODEL_DEFAULTS", "VERB_ALL", "VERB_DEEP", "VERB_NONE",
        "VERB_SIMPLE", "VERB_SYSTEM", "VERB_VERIFY",
    ]
