"""Tests for the ``orbit`` module."""

import numpy as np
import pytest

import pryngles as pr


def test_orbody_construction():
    """An ``Orbody`` stores its name, parent, and orbital elements."""
    S = pr.Orbody(m=1)
    assert S.name == "body"  # Default name is "body"
    assert S.parent is None
    assert S.m == 1
    assert S.elements == {"m": 1}

    P = pr.Orbody(name="Planet", parent=S, m=0.1, a=1, e=0.5)
    assert P.name == "Planet"
    assert P.parent is S
    assert P.m == 0.1
    assert P.a == 1
    assert P.e == 0.5
    assert P.elements == {"m": 0.1, "a": 1, "e": 0.5}


def test_orbody_invalid_element():
    """An invalid orbital element raises a ``ValueError``."""
    with pytest.raises(ValueError):
        pr.Orbody(m=1, not_an_element=1)


def test_orbit_construction():
    """An ``Orbit`` stores masses and orbital elements."""
    O = pr.Orbit(m1=1, m2=1, a=1, e=0.7, M=0)
    assert O.m1 == 1
    assert O.m2 == 1
    assert O.Mtot == 2
    assert O.a == 1
    assert O.e == 0.7


def test_orbit_invalid_mass():
    """An invalid mass type raises a ``ValueError``."""
    with pytest.raises(ValueError):
        pr.Orbit(m1="bad", m2=1)


def test_orbit_invalid_element():
    """An invalid orbital element raises a ``ValueError``."""
    with pytest.raises(ValueError):
        pr.Orbit(m1=1, m2=1, not_an_element=1)


def test_orbit_calculate_and_states():
    """``calculate_orbit`` and ``get_states`` produce particle states."""
    O = pr.Orbit(m1=1, m2=1e-3, a=0.5, e=0.4)
    O.calculate_orbit()
    sim, states = O.get_states()
    assert len(states) == 2
    # The two bodies have the expected masses.
    assert states[0]["m"] == 1
    assert states[1]["m"] == 1e-3
    # The relative separation should be ~a*(1-e)=0.3.
    r = np.array([states[1]["x"], states[1]["y"], states[1]["z"]]) - \
        np.array([states[0]["x"], states[0]["y"], states[0]["z"]])
    np.testing.assert_allclose(np.linalg.norm(r), 0.3, rtol=1e-6)


def test_orbit_hierarchical():
    """Nested ``Orbit`` objects can be assembled and calculated."""
    S1 = pr.Orbit(name="system1", m1=1, m2=1, a=1, e=0.7, M=0)
    S2 = pr.Orbit(name="system2", m1=1, m2=1, a=1, e=0, M=0)
    S3 = pr.Orbit(name="system3", m1=S1, m2=S2, a=5, e=0)
    S4 = pr.Orbit(name="system4", m1=S3, m2=1, a=20, e=0, E=90 * pr.Consts.deg)
    S4.calculate_orbit()
    sim, states = S4.get_states()

    assert len(states) == 5

    # The states are deterministic; compare against the reference values.
    expected = [
        {'m': 1.0, 'x': -2.65, 'y': -4.0, 'z': 0.0, 'vx': 0.1, 'vy': -2.1304644185603037, 'vz': 0.0},
        {'m': 1.0, 'x': -2.35, 'y': -4.0, 'z': 0.0, 'vx': 0.1, 'vy': 1.236037227560388, 'vz': 0.0},
        {'m': 1.0, 'x': 2.0, 'y': -4.0, 'z': 0.0, 'vx': 0.1, 'vy': -0.2598931856865896, 'vz': 0.0},
        {'m': 1.0, 'x': 3.0, 'y': -4.0, 'z': 0.0, 'vx': 0.1, 'vy': 1.1543203766865053, 'vz': 0.0},
        {'m': 1.0, 'x': 0.0, 'y': 16.0, 'z': 0.0, 'vx': -0.4, 'vy': 2.449293598294706e-17, 'vz': 0.0}
    ]
    for state, exp in zip(states, expected):
        assert state["m"] == exp["m"]
        for key in ("x", "y", "z", "vx", "vy", "vz"):
            np.testing.assert_allclose(state[key], exp[key], rtol=1e-6, atol=1e-12)


def test_build_tree():
    """``build_tree`` builds a nested tree and skips rings."""
    S = pr.Star()
    P = pr.Planet(parent=S)
    M = pr.Planet(parent=P)
    R = pr.Ring(parent=P)

    tree = pr.OrbitUtil.build_tree(S)
    # The tree is [S, [P, M]] (the ring is skipped).
    assert tree[0] is S
    assert tree[1][0] is P
    assert tree[1][1] is M


def test_build_system():
    """``build_system`` builds an ``Orbit`` from an orbital tree."""
    S = pr.Star(m=3)
    P = pr.Planet(parent=S, m=1, a=1, e=0.2)

    tree = pr.OrbitUtil.build_tree(S)
    orbit, pelements = pr.OrbitUtil.build_system(tree, units=["au", "msun", "yr"])
    orbit.calculate_orbit()
    sim, states = orbit.get_states()
    assert len(states) == 2

    expected = [
        {'m': 3.0, 'x': -0.2, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': -3.8475768228866953, 'vz': 0.0}, 
        {'m': 1.0, 'x': 0.6, 'y': 0.0, 'z': 0.0, 'vx': 0.0, 'vy': 11.542730468660086, 'vz': 0.0}
    ]
    for state, exp in zip(states, expected):
        assert state["m"] == exp["m"]
        for key in ("x", "y", "z", "vx", "vy", "vz"):
            np.testing.assert_allclose(state[key], exp[key], rtol=1e-6, atol=1e-12)
