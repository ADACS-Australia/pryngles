"""Regression tests for the quickstart_system.py pipeline.

Each test captures one part of the deterministic output (spangler state,
lightcurve, system metadata) and compares it against a stored golden
reference using pytest-regressions. The system is built once per module by
the module-scoped ``system`` fixture below, so the expensive lightcurve
computation is shared by all tests in this file.

Run with:
    uv run pytest tests/regression/test_quickstart_system.py

On first run the golden files are generated next to this test (in the
``test_quickstart_system/`` data directory). Regenerate them with:
    uv run pytest tests/regression/test_quickstart_system.py --force-regen
"""
import numpy as np
import pytest

import spiceypy as spy

import pryngles as pr

from tests.regression.utils import SystemChecks

_N_TIMES = 11


class TestQuickstartSystem(SystemChecks):

    @pytest.fixture(scope="class")
    def system(self):
        """Build the star/planet/ring system and compute its lightcurve.

        Module-scoped so the lightcurve is computed once and shared by all
        tests in this file.
        """
        system = pr.System()

        star = system.add(
            kind='Star',
            radius=pr.Consts.rsun / system.ul,
            limb_coeffs=[0.65],
        )

        planet = system.add(
            kind='Planet',
            parent=star,
            a=0.2,
            e=0.0,
            radius=pr.Consts.rsaturn / system.ul,
        )

        ring = system.add(
            kind='Ring',
            parent=planet,
            fi=1.5,
            fe=2.5,
            i=30 * pr.Consts.deg,
        )

        inc = 90.0
        omega = 0.0
        system.n_obs = spy.eul2m(np.deg2rad(omega), np.deg2rad(inc), 0, 3, 1, 3)[0]

        system.initialize_simulation()
        system.spangle_system()

        period_days = 365.25 * (planet.a ** 1.5)
        times_days = np.linspace(0.0, period_days, _N_TIMES)
        times_system = times_days * pr.Consts.day / system.ut

        system.compute_lightcurve(
            times=times_system,
            effects=['polarization'],
        )

        return system
