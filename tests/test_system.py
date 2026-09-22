import numpy as np
import pytest

import pryngles as pr


def test_init_default():
    """A default System has no bodies and canonical units."""
    sys = pr.System()
    assert sys.nbodies == 0
    assert sys.root is None
    assert sys.sg is None
    assert not sys._simulated
    assert not sys._spangled
    assert sys.units == ["au", "msun", "yr2pi"]


def test_init_units():
    """Can update units at initialization and after creation."""
    # Units are converted to SI units internally
    sys = pr.System(units=["au", "msun", "yr2pi"])
    assert sys.units == ["au", "msun", "yr2pi"]
    assert sys.ul == pr.Consts.au
    assert sys.um == pr.Consts.msun
    assert sys.ut == pr.Consts.yr2pi

    sys.update_units(["m", "kg", "s"])
    assert sys.units == ["m", "kg", "s"]
    assert sys.ul == 1.0
    assert sys.um == 1.0
    assert sys.ut == 1.0


def test_init_invalid_units():
    """An unrecognized unit raises ValueError."""
    sys = pr.System()
    with pytest.raises(ValueError):
        sys.update_units(["bad", "kg", "s"])


def test_add_defaults():
    """add() creates a Star root and assigns sources to children."""
    sys = pr.System()
    S = sys.add(m=8, radius=4)
    P = sys.add("Planet", parent=S, radius=2, a=10)
    assert sys.nbodies == 2
    assert sys.root.name == S.name
    # A planet's source is the root star
    assert P.source.name == S.name


def test_get_source():
    """_get_source() gets the source body for a body in a given system."""
    sys = pr.System()
    S = sys.add(m=8, radius=4)
    P = sys.add("Planet", parent=S, radius=2, a=10)
    R = sys.add("Ring", parent=P, radius=2, fi=1.3, fe=2.3)
    assert sys._get_source(S) == S
    assert sys._get_source(P) == S
    assert sys._get_source(R) == S


def test_add_duplicate_name():
    """Adding a body with an existing name raises ValueError."""
    sys = pr.System()
    S = sys.add(m=8, radius=4)
    P = sys.add("Planet", parent=S, radius=2, a=10)
    with pytest.raises(ValueError):
        sys.add("Planet", name=P.name, parent=S, radius=2, a=10)


def test_add_invalid_kind():
    """Adding a body with an unknown kind raises ValueError."""
    sys = pr.System()
    with pytest.raises(ValueError):
        sys.add("Foo")


def test_add_second_root():
    """Adding a second root (no parent) raises ValueError."""
    sys = pr.System()
    sys.add(m=8, radius=4)
    with pytest.raises(ValueError):
        sys.add("Star", name="Star2", m=8, radius=4)


def test_remove():
    """remove() deletes a body and its children from the system."""
    sys = pr.System()
    S = sys.add(name="Star", m=8, radius=4)
    P = sys.add("Planet", parent=S, name="Planet", radius=2, a=10)
    M = sys.add("Planet", parent=P, name="Moon", radius=2, a=1)
    R = sys.add("Ring", parent=P, name="Ring", fi=1.3, fe=2.3)
    assert sys.nbodies == 4

    # Removing the planet also removes its children (Moon, Ring)
    sys.remove("Planet")
    assert sys.nbodies == 1
    assert "Planet" not in sys.bodies
    assert "Moon" not in sys.bodies
    assert "Ring" not in sys.bodies


def test_remove_missing():
    """Removing a non-existent body raises ValueError."""
    sys = pr.System()
    S = sys.add(name="Star", m=8, radius=4)
    with pytest.raises(ValueError):
        sys.remove("Planet")


def test_update_body_before_spangle():
    """update_body modifies a body's properties before spangling."""
    sys = pr.System()
    S = sys.add("Star", name="Star", m=8, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", radius=0.2, a=2)
    assert P.radius == 0.2
    sys.update_body(P, radius=0.5)
    assert P.radius == 0.5


def test_update_body_after_spangle():
    """update_body raises AssertionError after the system is spangled."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=8, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    with pytest.raises(AssertionError):
        sys.update_body("Planet", fe=3.0)


def test_spangle_flow():
    """initialize_simulation + spangle_system produces a joined Spangler."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=9, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    assert sys._spangled
    assert sys.sg is not None
    assert sys.sg.shape == "Join"
    assert len(sys.sg.data) == len(S.sg.data) + len(P.sg.data)
    assert set(sys.sg.data.name.unique()) == {"Star", "Planet"}


def test_set_observer_before_spangle():
    """_set_observer raises AssertionError before the system is spangled."""
    sys = pr.System()
    S = sys.add(nspangles=100, m=8, radius=1)
    P = sys.add("Planet", parent=S, nspangles=100, m=1, radius=0.2, a=5)
    with pytest.raises(AssertionError):
        sys._set_observer(nvec=[1, 0, 0])


def test_spangle_before_simulation():
    """spangle_system raises AssertionError before initialize_simulation."""
    sys = pr.System()
    S = sys.add(nspangles=100, m=8, radius=1)
    P = sys.add("Planet", parent=S, nspangles=100, m=1, radius=0.2, a=5)
    with pytest.raises(AssertionError):
        sys.spangle_system()


def test_integrate_before_spangle():
    """integrate raises AssertionError before the system is spangled."""
    sys = pr.System()
    S = sys.add("Star", name="Star", m=8, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", radius=0.2, a=2)
    with pytest.raises(AssertionError):
        sys.integrate(10)


def test_set_luz_before_observer():
    """_set_luz raises AssertionError if the observer has not been set."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=9, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    # spangle_system sets the observer via update_perspective; reset the flag
    sys._observer_set = False
    with pytest.raises(AssertionError):
        sys._set_luz()


def test_set_observer_and_luz():
    """_set_observer then _set_luz sets the observer and light flags."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=9, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    sys._set_observer(nvec=[0, 0, 1])
    sys._set_luz()
    assert sys._observer_set
    assert sys._luz_set


def test_update_perspective():
    """update_perspective sets the observer direction and flags."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=9, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    sys.update_perspective(n_obs=[1, 0, 0])
    np.testing.assert_allclose(sys.n_obs, [1, 0, 0])
    assert sys._observer_set
    assert sys._luz_set


def test_integrate():
    """integrate advances the simulation and updates body centers."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=1, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, m=1e-3, a=5)
    sys.initialize_simulation()
    sys.spangle_system()

    t_before = sys.sim.t
    center_before = np.array(sys.sim.particles[P.rbhash].xyz)

    sys.integrate(10)

    assert sys.sim.t > t_before
    # The planet should have moved from its initial position
    center_after = np.array(sys.sim.particles[P.rbhash].xyz)
    assert not np.allclose(center_before, center_after)
    # The body's center and the spangler's center_ecl column are updated
    np.testing.assert_allclose(P.center_ecl, center_after)
    np.testing.assert_allclose(
        np.array(sys.sg.data.loc[sys.sg.data.name == "Planet", "center_ecl"].iloc[0]),
        center_after,
    )


def test_integrate_perspective():
    """integrate_perspective advances time and updates the observer."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=1, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, m=1e-3, a=5)
    sys.initialize_simulation()
    sys.spangle_system()
    t_before = sys.sim.t
    sys.integrate_perspective(10, n_obs=[1, 0, 0])
    assert sys.sim.t > t_before
    np.testing.assert_allclose(sys.n_obs, [1, 0, 0])


def test_update_scatterers():
    """spangle_system assigns a scatterer to every spangle."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=9, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    assert "scatterer" in sys.sg.data.columns
    assert (sys.sg.data.scatterer != "").all()


def test_update_optical_depth():
    """spangle_system sets the tau_gray_optical column."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=9, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    assert (sys.sg.data["tau_gray_optical"] == np.inf).all()
    sys.bodies["Planet"].tau_gray_optical = 0.5
    sys._update_optical_depth()
    assert (sys.sg.data["tau_gray_optical"] == 0.5).all


def test_update_albedos():
    """_update_albedos computes directional albedo per spangle."""
    sys = pr.System()
    S = sys.add("Star", name="Star", nspangles=100, m=9, radius=1)
    P = sys.add("Planet", parent=S, name="Planet", nspangles=100, radius=0.2, a=2)
    sys.initialize_simulation()
    sys.spangle_system()
    sys._update_albedos()

    data = sys.sg.data
    assert "lambertian_albedo" in data.columns

    # Stellar spangles have zero albedo (they emit, they don't reflect)
    assert (data.loc[data.name == "Star", "lambertian_albedo"] == 0).all()

    # Spangles not facing the light source have zero albedo
    assert (data.loc[data.cos_luz < 0, "lambertian_albedo"] == 0).all()

    # Albedo is bounded in [0, 1]
    assert (data.lambertian_albedo >= 0).all()
    assert (data.lambertian_albedo <= 1).all()

    # For a Lambertian surface with AL=1, albedo is ~1 at normal incidence
    planet = data.loc[data.name == "Planet"]
    cond = planet.cos_luz > 0.99
    assert cond.any()
    np.testing.assert_allclose(planet.loc[cond, "lambertian_albedo"], 1.0, atol=1e-3)


@pytest.mark.skip(reason="Not yet implemented")
def test_update_visibility_state():
    """update_visibility_state applies occlusion to the visible state."""
    # TODO: build a multi-body system (e.g. star + planet), set the observer,
    # call update_visibility_state, and assert that spangles behind the planet
    # are no longer visible (visible == intersect).
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_update_illumination_state():
    """update_illumination_state applies shadowing to the illuminated state."""
    # TODO: build a multi-body system, set observer and light source, call
    # update_illumination_state, and assert that shadowed spangles have
    # illuminated == False and shadow == True.
    pass
