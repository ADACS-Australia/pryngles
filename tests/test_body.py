"""Tests for the ``body`` module."""

import numpy as np
import pytest

import pryngles as pr


def test_body_construction():
    """A ``Body`` can be constructed with defaults and updated properties."""
    B = pr.Body("Body", pr.BODY_DEFAULTS, None, m=2, x=2, a=1, name_by_kind=True)
    assert B.kind == "Body"
    assert B.name == "Body"
    assert B.m == 2
    assert B.x == 2
    assert B.a == 1
    # ``elements`` holds the orbital properties.
    assert B.elements["m"] == 2
    assert B.elements["x"] == 2
    assert B.elements["a"] == 1


def test_body_update():
    """``update_body`` updates properties and rejects unknown ones."""
    B = pr.Body("Body", pr.BODY_DEFAULTS, None, name="B")
    B.update_body(name="B2")
    assert B.name == "B2"
    with pytest.raises(ValueError):
        B.update_body(not_a_property=1)


def test_body_parent_child():
    """A child body is registered in the parent's ``childs``."""
    B = pr.Body("Body", pr.BODY_DEFAULTS, None, name="B")
    C = pr.Body("Body", pr.BODY_DEFAULTS, B, name="C")
    assert C.parent is B
    assert B.childs["C"] is C


def test_body_invalid_parent():
    """A non-``Body`` parent raises an ``AssertionError``."""
    with pytest.raises(AssertionError):
        pr.Body("Body", pr.BODY_DEFAULTS, "Nada")


def test_body_legacy_primary():
    """The legacy ``primary`` argument is used as the parent."""
    B = pr.Body("Body", pr.BODY_DEFAULTS, None, name="B")
    C = pr.Body("Body", pr.BODY_DEFAULTS, None, name="C", primary=B)
    assert C.parent is B


def test_body_derived_properties():
    """``wrot`` and ``n_equ`` are derived from the body properties."""
    B = pr.Body("Body", pr.BODY_DEFAULTS, None, name="B")
    np.testing.assert_allclose(B.wrot, 2 * np.pi / pr.BODY_DEFAULTS["prot"], rtol=1e-7)
    # Default rotation axis points along +z.
    np.testing.assert_allclose(B.n_equ, [0.0, 0.0, 1.0], atol=1e-12)


def test_body_spangle():
    """``spangle_body`` creates a populated ``Spangler``."""
    B = pr.Body("Body", pr.BODY_DEFAULTS, None, name="B", nspangles=100)
    B.spangle_body()
    assert B.sg is not None
    assert len(B.sg.data) == 100
    assert (B.sg.data["name"] == "B").all()


def test_star():
    """A ``Star`` can be created and updated."""
    S = pr.Star()
    assert S.kind == "Star"

    S.update_star(m=2)
    assert S.m == 2


def test_star_invalid_parent():
    """A ``Star`` parent must be another ``Star``."""
    B = pr.Body("Body", pr.BODY_DEFAULTS, None, name="B")
    
    with pytest.raises(ValueError):
        pr.Star(parent=B)

    S1 = pr.Star()
    S2 = pr.Star(parent=S1)

    assert S2.parent is S1


def test_update_star():
    """A ``Star`` updates limb darkening when coefficients are updated."""
    S = pr.Star()
    assert S.kind == "Star"
    assert S.limb_coeffs == []
    np.testing.assert_allclose(S.norm_limb_darkening, 3.141592653589793, rtol=1e-10)

    S.update_star(limb_coeffs=[1, 1])
    assert S.limb_coeffs == [1, 1]
    np.testing.assert_allclose(S.norm_limb_darkening, 1.5707963267948863, rtol=1e-10)


def test_star_spangle():
    """``spangle_body`` creates a populated ``Spangler`` for a ``Star``."""
    S = pr.Star(nspangles=100)
    S.spangle_body()
    assert S.sg is not None
    assert len(S.sg.data) == 100
    assert (S.sg.data["source"] == True).all()
    assert (S.sg.data["name"] == S.name).all()


def test_planet():
    """A ``Planet`` requires a parent and can be updated."""
    S = pr.Star()
    with pytest.raises(ValueError):
        pr.Planet()
    P = pr.Planet(parent=S)
    assert P.kind == "Planet"
    
    P.update_planet(vz=0.2)
    assert P.vz == 0.2


def test_planet_invalid_parent():
    """A ``Planet`` parent must be a ``Body``."""
    with pytest.raises(AssertionError):
        pr.Planet(parent="Nada")

    S = pr.Star()
    P = pr.Planet(parent=S)

    assert P.parent is S


def test_planet_set_temperature_model():
    """``set_temperature_model`` validates the model and stores it."""
    S = pr.Star()
    P = pr.Planet(parent=S)

    # Not a dict.
    with pytest.raises(TypeError):
        P.set_temperature_model("Uniform Temperature")

    # Unknown model type.
    with pytest.raises(ValueError):
        P.set_temperature_model({"type": "Unknown", "params": {}})

    # Missing mandatory parameters.
    with pytest.raises(ValueError):
        P.set_temperature_model({"type": "Uniform Temperature", "params": {}})

    # Valid model.
    P.set_temperature_model({"type": "Uniform Temperature", "params": {"T_planet": 1500}})
    assert P.T_model == {"type": "Uniform Temperature", "params": {"T_planet": 1500}}


def test_planet_update_temperature_requires_spangler():
    """``update_temperature`` raises if the body is not spangled."""
    S = pr.Star()
    P = pr.Planet(parent=S)
    P.set_temperature_model({"type": "Uniform Temperature", "params": {"T_planet": 1500}})
    with pytest.raises(RuntimeError):
        P.update_temperature()


def test_planet_update_temperature_requires_model():
    """``update_temperature`` raises if no temperature model is set."""
    S = pr.Star()
    P = pr.Planet(parent=S)
    P.spangle_body()
    with pytest.raises(RuntimeError):
        P.update_temperature()


def test_planet_uniform_temperature():
    """The uniform temperature model sets a constant ``Tem``."""
    S = pr.Star()
    P = pr.Planet(parent=S, nspangles=100)
    P.spangle_body()
    P.set_temperature_model({"type": "Uniform Temperature", "params": {"T_planet": 1500}})
    P.update_temperature()
    assert (P.sg.data["Tem"] == 1500).all()


def test_planet_two_temperature():
    """The two-temperature model sets day/night temperatures by ``cos_luz``."""
    S = pr.Star()
    P = pr.Planet(parent=S, nspangles=100)
    P.spangle_body()
    P.set_temperature_model(
        {"type": "Two Temperature", "params": {"T_day": 2000, "T_night": 500}}
    )
    P.update_temperature()

    cond_day = (P.sg.data.cos_luz > 0)
    cond_night = ~cond_day
    assert (P.sg.data.loc[cond_day, "Tem"] == 2000).all()
    assert (P.sg.data.loc[cond_night, "Tem"] == 500).all()


def test_planet_zhang_showman_temperature():
    """The Zhang-Showman model produces a finite temperature map."""
    S = pr.Star()
    P = pr.Planet(parent=S, nspangles=100)
    # The model needs a non-zero light-source direction, so position the
    # planet away from the star and use numpy arrays for the centers.
    P.center_ecl = np.array([1.0, 0.0, 0.0])
    P.root.center_ecl = np.array([0.0, 0.0, 0.0])
    P.spangle_body()
    P.set_temperature_model(
        {"type": "Zhang-Showman", "params": {"xi_ratio": 0.3, "T_night": 1200, "Delta_T": 800}}
    )
    P.update_temperature()
    assert np.isfinite(P.sg.data["Tem"]).all()

    assert (P.sg.data["Tem"].max() < 2000) and (P.sg.data["Tem"].min() > 1200)
    # Test the first 10 values against a reference to ensure the model is producing consistent results.
    expected = [
        1200.003906,
        1385.599360,
        1201.213468,
        1266.800000,
        1369.730932,
        1200.083443,
        1536.865786,
        1213.530980,
        1200.004807,
        1606.494474,
    ]
    np.testing.assert_allclose(P.sg.data["Tem"].iloc[:10].to_numpy(), expected, rtol=1e-6)


def test_ring():
    """A ``Ring`` requires a parent and updates its radii."""
    S = pr.Star()
    P = pr.Planet(parent=S)
    with pytest.raises(ValueError):
        pr.Ring()
    R = pr.Ring(parent=P)
    assert R.kind == "Ring"
    R.update_ring(fe=3)
    assert R.fe == 3
    # Inner/outer radii are derived from the parent radius.
    assert R.ri == R.fi * P.radius
    assert R.re == R.fe * P.radius


def test_update_ring():
    """A ``Ring`` updates its inner/outer radii when the parent radius changes."""
    S = pr.Star()
    P = pr.Planet(parent=S, radius=1)
    R = pr.Ring(parent=P, fi=1.5, fe=2.5)
    assert R.ri == 1.5
    assert R.re == 2.5
    P.update_planet(radius=2)
    R.update_ring(fi=2, fe=3)
    assert R.fi == 2
    assert R.fe == 3
    assert R.ri == 4.0
    assert R.re == 6.0


def test_ring_spangle():
    """A ``Ring`` can be spangled."""
    S = pr.Star()
    P = pr.Planet(parent=S)
    R = pr.Ring(parent=P, nspangles=100)
    R.spangle_body()
    assert R.sg is not None
    assert len(R.sg.data) > 0


def test_detector_defaults():
    """A ``Detector`` is initialized with the default properties."""
    D = pr.Detector()
    assert D.wavelength_min == pr.DETECTOR_PROPERTIES["wavelength_min"]
    assert D.wavelength_max == pr.DETECTOR_PROPERTIES["wavelength_max"]
    assert D.apperture == pr.DETECTOR_PROPERTIES["apperture"]
    assert D.quantum_eff == pr.DETECTOR_PROPERTIES["quantum_eff"]
    assert D.t_cadence == pr.DETECTOR_PROPERTIES["t_cadence"]
    assert D.distance == pr.DETECTOR_PROPERTIES["distance"]


def test_detector_update():
    """A ``Detector`` accepts valid properties and rejects invalid ones."""
    D = pr.Detector(apperture=1.0, quantum_eff=0.5)
    assert D.apperture == 1.0
    assert D.quantum_eff == 0.5
    with pytest.raises(AssertionError):
        pr.Detector(not_a_property=1)


def test_detector_set_source():
    """``set_source`` computes the normal flux and validates the source."""
    D = pr.Detector()
    S = pr.Star()
    S.ul = pr.Consts.au  # canonical length unit (set by System normally)
    D.set_source(S)
    np.testing.assert_allclose(D.normal_flux, 2098153.848140682, rtol=1e-6)

    P = pr.Planet(parent=S)
    with pytest.raises(ValueError):
        D.set_source(P)


def test_detector_generate_signal():
    """``generate_signal`` produces a sorted signal with errors."""
    D = pr.Detector(t_cadence=10)
    S = pr.Star()
    S.ul = pr.Consts.au  # canonical length unit (set by System normally)
    D.set_source(S)

    np.random.seed(42)
    times = np.linspace(0, 100, 50)
    fluxes = np.ones_like(times)
    signal_times, signal_lc, signal_error = D.generate_signal(times, fluxes)

    assert len(signal_times) == len(signal_lc) == len(signal_error)
    assert np.all(np.diff(signal_times) >= 0)
    assert (signal_error > 0).all()

    # Deterministic values with the fixed seed.
    np.testing.assert_allclose(
        signal_times[:5],
        [-4.69584628, -2.29007168, -0.22732829, 1.25459881, 1.33638157],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        signal_lc[:5],
        [1.00032107, 1.00018238, 0.99995532, 1.00049975, 0.9999731],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        signal_error[:5],
        [0.00021835, 0.00021833, 0.00021831, 0.00021837, 0.00021831],
        atol=1e-8,
    )
