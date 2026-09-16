"""Regression tests for the wasp43b_system.py pipeline.

This is a more involved example than the quickstart: it builds a
WASP-43b-like star/planet/ring system with a temperature model and a
detector, and computes three separate lightcurves (transit, emission,
polarization). The detector simulates a stochastic signal, so the RNG is
seeded before generating it to keep the golden file reproducible.

Each test captures one part of the deterministic output and compares it
against a stored golden reference using pytest-regressions. The system is
built once per class by the class-scoped ``system`` fixture below, so
the expensive lightcurve computations are shared by all tests in this
file.

Run with:
    uv run pytest tests/regression/test_wasp43b_system.py

On first run the golden files are generated next to this test (in the
``test_wasp43b_system/`` data directory). Regenerate them with:
    uv run pytest tests/regression/test_wasp43b_system.py --force-regen
"""
import numpy as np
import pytest

import spiceypy as spy

import pryngles as pr

from tests.regression.utils import SystemChecks


# WASP-43 system parameters.
_PERIOD = 0.81347753          # Orbital period [days]
_A_ABS = 0.01526              # Absolute semi-major axis [AU]
_A = 4.857                    # Semi-major axis [R_star]
_INC = 90                     # Orbital inclination [deg]
_ECC = 0.0                    # Eccentricity
_OMEGA = 0.0                  # Argument of periastron [deg]
_M_STAR = 0.717               # Stellar mass [M_sun]
_R_STAR = 0.667               # Stellar radius [R_sun]
_T_STAR = 4520                # Stellar effective temperature [K]
_M_PLANET = 0                 # Planet mass [M_jup] (no mass)
_R_PLANET = 1.036             # Planet radius [R_jup]
_T_NIGHT = 1200               # Nightside temperature [K]
_DELTA_T = 800                # Day-night temperature contrast [K]
_XI = 0.3                     # Hot-spot offset parameter
_R_IN = 1.5                   # Inner ring radius [R_planet]
_R_OUT = 2.5                  # Outer ring radius [R_planet]
_RING_INC = 60                # Ring inclination w.r.t. orbital plane [deg]
_RING_TAU = 0.4               # Ring optical depth

# Detector properties (passed to compute_lightcurve as the signal dict).
_DETECTOR_PROPERTIES = {
    'wavelength_min': 1.1e-6,
    'wavelength_max': 1.7e-6,
    'apperture': 0.5,
    'quantum_eff': 0.9,
    't_cadence': 15 * 60,
    'distance': 100 * pr.Consts.pc,
}

_N_TIMES = 11

class TestWasp43bSystem(SystemChecks):

    @pytest.fixture(scope="class")
    def system(self):
        """Build the WASP-43b system and compute its three lightcurves.

        Class-scoped so the expensive lightcurve computations are shared by
        all tests in this class. The RNG is seeded before the detector
        signal is generated so the stochastic signal is reproducible.
        """
        R_star_AU = _A_ABS / _A
        R_star_m = R_star_AU * pr.Consts.au
        M_star = _M_STAR * pr.Consts.msun
        M_planet = _M_PLANET * pr.Consts.mjupiter
        R_planet = _R_PLANET * pr.Consts.rjupiter

        system = pr.System()

        star = system.add(
            kind="Star",
            m=M_star / system.um,
            T_eff=_T_STAR,
            radius=R_star_m / system.ul,
            limb_coeffs=[0.65],
        )

        planet = system.add(
            kind="Planet",
            parent=star,
            m=M_planet / system.um,
            radius=R_planet / system.ul,
            a=_A_ABS,
            inc=pr.DEG * (90 - _INC),
            e=_ECC,
            omega=pr.DEG * _OMEGA,
        )

        ring = system.add(
            kind='Ring',
            parent=planet,
            fi=_R_IN,
            fe=_R_OUT,
            i=pr.DEG * _RING_INC,
            taur=_RING_TAU,
        )

        system.n_obs = spy.eul2m(
            np.deg2rad(_OMEGA),
            np.deg2rad(_INC),
            0,
            3, 1, 3,
        )[0]

        system.initialize_simulation()
        system.spangle_system()

        # Planet temperature model (Zhang-Showman).
        planet.set_temperature_model({
            "type": "Zhang-Showman",
            "params": {
                "T_night": _T_NIGHT,
                "Delta_T": _DELTA_T,
                "xi_ratio": _XI,
            },
        })

        # Integration times: one orbital period centered on transit.
        times_days = np.linspace(-_PERIOD / 2, _PERIOD / 2, _N_TIMES)
        times_system = times_days * pr.Consts.day / system.ut

        # Transit + detector signal (seed RNG for a reproducible signal).
        np.random.seed(42)
        system.compute_lightcurve(
            times=times_system,
            effects=['transit', 'emission', 'polarization'],
            signal=_DETECTOR_PROPERTIES,
            bandwidth=(_DETECTOR_PROPERTIES['wavelength_min'], _DETECTOR_PROPERTIES['wavelength_max']),
        )

        return system
