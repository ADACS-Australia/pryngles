"""Tests for the functions in the ``scatterer`` module."""

import numpy as np
import pytest

import pryngles as pr


@pytest.fixture(autouse=True)
def _reset_catalogue():
    """Reset the shared scatterer catalogue before and after each test."""
    pr.Scatterer.reset_catalogue()
    yield
    pr.Scatterer.reset_catalogue()


def test_neutral_surface_albedo():
    """``NeutralSurface`` returns an albedo of 1 for any input."""
    s = pr.NeutralSurface()
    assert s.get_albedo(0.5, 0.3, 0.0, 0.0) == 1
    assert s.get_albedo(0.0, 0.0, 0.0, 0.0) == 1
    assert s.get_albedo(1.0, 1.0, 1.0, 1.0) == 1


def test_blackbody_surface_albedo():
    """``BlackBodySurface`` returns an albedo of 0 for any input."""
    s = pr.BlackBodySurface()
    assert s.get_albedo(0.5, 0.3, 0.0, 0.0) == 0
    assert s.get_albedo(0.0, 0.0, 0.0, 0.0) == 0
    assert s.get_albedo(1.0, 1.0, 1.0, 1.0) == 0


def test_register_populates_catalogue():
    """Creating a scatterer registers it in the global catalogue."""
    assert len(pr.SCATTERERS_CATALOGUE) == 0
    pr.NeutralSurface()
    assert len(pr.SCATTERERS_CATALOGUE) == 1
    assert any(isinstance(s, pr.NeutralSurface) for s in pr.SCATTERERS_CATALOGUE.values())
    assert any(s.params["name"] == "NeutralSurface" for s in pr.SCATTERERS_CATALOGUE.values())


def test_register_reuses_existing_entry():
    """Creating a scatterer with the same params reuses the catalogue entry."""
    s1 = pr.NeutralSurface()
    s2 = pr.NeutralSurface()
    # Same params -> same hash -> same catalogue key.
    assert s1.hash == s2.hash
    assert len(pr.SCATTERERS_CATALOGUE) == 1


def test_register_distinct_params_distinct_entries():
    """Scatterers with different params get different catalogue entries."""
    s1 = pr.LambertianGraySurface(AL=0.3)
    s2 = pr.LambertianGraySurface(AL=0.7)
    assert s1.hash != s2.hash
    assert len(pr.SCATTERERS_CATALOGUE) == 2


def test_reset_catalogue_clears():
    """``reset_catalogue`` empties the global catalogue."""
    pr.NeutralSurface()
    assert len(pr.SCATTERERS_CATALOGUE) == 1
    pr.Scatterer.reset_catalogue()
    assert len(pr.SCATTERERS_CATALOGUE) == 0


def test_lambertian_surface_albedo():
    """``LambertianGraySurface`` albedo stays close to the specified ``AL``."""
    AL = 0.5
    s = pr.LambertianGraySurface(AL=AL)
    for eta in np.linspace(0.1, 1.0, 10):
        np.testing.assert_allclose(
            s.get_albedo(eta, 0.5, 0.0, 0.0), AL, atol=1e-3
        )


def test_lambertian_surface_al_controls_max():
    """A higher ``AL`` gives a higher albedo at normal incidence."""
    s_low = pr.LambertianGraySurface(AL=0.3)
    s_high = pr.LambertianGraySurface(AL=0.7)
    assert s_high.get_albedo(1.0, 0.5, 0.0, 0.0) > s_low.get_albedo(1.0, 0.5, 0.0, 0.0)


def test_lambertian_surface_custom_phase_law():
    """A custom phase law is used in the albedo computation."""
    AL = 0.5
    s = pr.LambertianGraySurface(
        AL=AL, phase_law=lambda eta, zeta, delta, lamb, params: eta**2
    )
    for eta in np.linspace(0.1, 1.0, 10):
        np.testing.assert_allclose(
            s.get_albedo(eta, 0.5, 0.0, 0.0), AL * eta, atol=1e-3
        )

def test_lambertian_atmosphere_reference_values():
    """``LambertianGrayAtmosphere`` reproduces the expected albedo values."""
    s = pr.LambertianGrayAtmosphere(AS=0.5)
    etas = np.linspace(0.1, 1.0, 10)
    expected = [
        0.647888, 0.611716, 0.581048, 0.554042, 0.530023,
        0.508227, 0.488222, 0.469920, 0.453186, 0.437472,
    ]
    for eta, ref in zip(etas, expected):
        np.testing.assert_allclose(
            s.get_albedo(eta, 0.5, 0.0, 0.0), ref, atol=1e-3
        )


def test_lambertian_atmosphere_conservative_scattering():
    """In the conservative-scattering limit (``AS=1``) the atmosphere is a
    perfect reflector, so the albedo is ~1 for all incidence angles."""
    s = pr.LambertianGrayAtmosphere(AS=1.0)

    assert s.gamma0 == 1
    for eta in np.linspace(0.1, 1.0, 10):
        np.testing.assert_allclose(
            s.get_albedo(eta, 0.5, 0.0, 0.0), 1.0, atol=1e-2
        )


def test_lambertian_atmosphere_decreasing():
    """``LambertianGrayAtmosphere`` albedo decreases with incidence cosine."""
    s = pr.LambertianGrayAtmosphere(AS=0.5)
    etas = np.linspace(0.1, 1.0, 10)
    albedos = [s.get_albedo(eta, 0.5, 0.0, 0.0) for eta in etas]
    assert all(b <= a for a, b in zip(albedos, albedos[1:]))


def test_lambertian_atmosphere_as_controls_max():
    """A higher ``AS`` gives a higher albedo at normal incidence."""
    s_low = pr.LambertianGrayAtmosphere(AS=0.3)
    s_high = pr.LambertianGrayAtmosphere(AS=0.7)
    assert s_high.get_albedo(1.0, 0.5, 0.0, 0.0) > s_low.get_albedo(1.0, 0.5, 0.0, 0.0)
