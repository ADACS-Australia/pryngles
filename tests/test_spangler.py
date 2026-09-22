import numpy as np
import pytest

import pryngles as pr


def test_const():
    """SPANGLER_KEY_ORDERING and SPANGLER_COLUMNS must contain the same keys."""
    for key in pr.SPANGLER_KEY_ORDERING:
        assert key in pr.SPANGLER_COLUMNS, f"Column '{key}' in SPANGLER_KEY_ORDERING not in SPANGLER_COLUMNS"
    for key in pr.SPANGLER_COLUMNS:
        assert key in pr.SPANGLER_KEY_ORDERING, f"Column '{key}' in SPANGLER_COLUMNS not in SPANGLER_KEY_ORDERING"


def test_init_basic():
    """A basic Spangler has the expected number of rows and default state."""
    sg = pr.Spangler(nspangles=3, center_equ=[0, 0, 0], n_equ=[1, 0, 0])
    assert sg.nspangles == 3
    assert len(sg.data) == 3
    assert sg.shape == "vanilla"
    # Default state: unset True, visibility/source states False
    assert (sg.data.unset == True).all()
    for col in list(pr.SPANGLER_VISIBILITY_STATES) + list(pr.SPANGLER_SOURCE_STATES):
        assert (sg.data[col] == False).all()


def test_init_join():
    """Joining spanglers combines data and sets shape to 'Join'."""
    sg1 = pr.Spangler(name="Body 1", nspangles=3, w=40 * pr.Consts.deg, n_equ=[1, 1, 0])
    sg2 = pr.Spangler(name="Body 2", nspangles=3, w=30 * pr.Consts.deg, n_equ=[1, 0, 1])
    sg = pr.Spangler(spanglers=[sg1, sg2])
    assert sg.shape == "Join"
    assert sg.nspangles == 6
    assert sg.name == ["Body 1", "Body 2"]
    assert set(sg.data.name.unique()) == {"Body 1", "Body 2"}


def test_join():
    """Joining spanglers combines data and sets shape to 'Join'."""
    sg1 = pr.Spangler(name="A", nspangles=10)
    sg1.populate_spangler(shape="sphere", scale=1, seed=1)
    sg2 = pr.Spangler(name="B", nspangles=20)
    sg2.populate_spangler(shape="sphere", scale=1, seed=1)
    sgj = pr.Spangler(spanglers=[sg1, sg2])
    assert sgj.shape == "Join"
    assert sgj.nspangles == 30
    assert sgj.name == ["A", "B"]
    assert set(sgj.data.name.unique()) == {"A", "B"}



def test_reset_state():
    """reset_state clears all visibility/source states and sets unset."""
    sg = pr.Spangler(nspangles=100)
    sg.populate_spangler(shape="sphere", scale=1, seed=1)
    sg.set_positions()
    sg.set_observer(nvec=[0, 0, 1])
    sg.set_luz(nvec=[1, 0, 0])
    # Some spangles should be visible/illuminated before reset
    assert sg.data.visible.any()
    assert sg.data.illuminated.any()

    sg.reset_state()
    assert (sg.data.unset == True).all()
    for col in list(pr.SPANGLER_VISIBILITY_STATES) + list(pr.SPANGLER_SOURCE_STATES):
        assert (sg.data[col] == False).all()
    for coords in "int", "obs", "luz":
        assert (sg.data["hidden_by_" + coords] == "").all()
        assert (sg.data["transit_over_" + coords] == "").all()


def test_set_scale():
    """set_scale scales lengths by scale, areas by scale**2, vectors by scale."""
    sg = pr.Spangler(nspangles=10, center_equ=[1, 2, 3])
    sg.populate_spangler(shape="circle", scale=1, seed=1)
    sg.set_positions()

    asp_before = sg.data.asp.iloc[0]
    x_before = sg.data.x_equ.iloc[0]
    center_before = np.array(sg.data.center_equ.iloc[0])

    scale = 3
    sg.set_scale(scale)

    assert sg.scale == scale
    np.testing.assert_allclose(sg.data.asp.iloc[0], asp_before * scale**2)
    np.testing.assert_allclose(sg.data.x_equ.iloc[0], x_before * scale)
    np.testing.assert_allclose(np.array(sg.data.center_equ.iloc[0]), center_before * scale)


def test_populate_spangler_sphere():
    """Sphere spangles lie on a sphere of radius scale with unit normals."""
    scale = 2
    sg = pr.Spangler(nspangles=100)
    sg.populate_spangler(shape="sphere", scale=scale, seed=1)
    sg.set_positions()
    r = np.linalg.norm(sg.data[["x_equ", "y_equ", "z_equ"]].values, axis=1)
    np.testing.assert_allclose(r, scale, atol=1e-6)
    ns = np.stack(sg.data.ns_equ.values)
    np.testing.assert_allclose(np.linalg.norm(ns, axis=1), 1, atol=1e-6)


def test_populate_spangler_circle():
    """Circle spangles lie in the equatorial plane."""
    sg = pr.Spangler(nspangles=50)
    sg.populate_spangler(shape="circle", scale=1, seed=1)
    np.testing.assert_allclose(sg.data.z_equ, 0, atol=1e-12)
    r = np.linalg.norm(sg.data[["x_equ", "y_equ", "z_equ"]].values, axis=1)
    assert r.max() <= 1.0 + 1e-12


def test_set_intersect_infinite():
    """set_intersect with center=None returns unit n_int and infinite d_int."""
    sg = pr.Spangler(nspangles=50)
    sg.populate_spangler(shape="sphere", scale=1, seed=1)
    sg.set_positions()
    cond, n_int, d_int = sg.set_intersect(nvec=[1, 0, 1], center=None)
    assert cond.all()
    np.testing.assert_allclose(np.linalg.norm(n_int), 1)
    assert np.isinf(d_int)
    np.testing.assert_allclose(n_int, [1, 0, 1] / np.sqrt(2))


def test_set_intersect_finite():
    """set_intersect with a finite center returns finite d_int."""
    sg = pr.Spangler(nspangles=50)
    sg.populate_spangler(shape="sphere", scale=1, seed=1)
    sg.set_positions()
    cond, n_int, d_int = sg.set_intersect(nvec=[0, 0, 1], center=[0, 0, 5])
    assert cond.all()
    np.testing.assert_allclose(d_int, 5)
    np.testing.assert_allclose(np.linalg.norm(n_int), 1)


def test_set_observer():
    """set_observer sets n_obs and marks visible spangles towards the observer."""
    sg = pr.Spangler(nspangles=200)
    sg.populate_spangler(shape="sphere", scale=1, seed=1)
    sg.set_positions()
    sg.set_observer(nvec=[0, 0, 1])
    np.testing.assert_allclose(sg.n_obs, [0, 0, 1])
    np.testing.assert_allclose(sg.rqf_obs, pr.Science.spherical([0, 0, 1]))
    # For a sphere with no hidden spangles, visible == cos_obs > 0
    assert (sg.data.visible == (sg.data.cos_obs > 0)).all()
    assert sg.data.visible.any()

    # Now check when the observer is at a finite distance
    sg.set_observer(nvec=[0, 0, 1], center=[0, 0, 5])
    np.testing.assert_allclose(sg.d_obs, 5)
    # For a sphere with no hidden spangles, visible == cos_obs > 0
    assert (sg.data.visible == (sg.data.cos_obs > 0)).all()


def test_set_luz():
    """set_luz sets n_luz and marks illuminated spangles towards the light source."""
    sg = pr.Spangler(nspangles=200)
    sg.populate_spangler(shape="sphere", scale=1, seed=1)
    sg.set_positions()
    sg.set_observer(nvec=[0, 0, 1])
    sg.set_luz(nvec=[1, 0, 0])
    np.testing.assert_allclose(sg.n_luz, [1, 0, 0])
    np.testing.assert_allclose(sg.rqf_luz, pr.Science.spherical([1, 0, 0]))
    # For a sphere with no hidden spangles, illuminated == cos_luz > 0
    assert (sg.data.illuminated == (sg.data.cos_luz > 0)).all()
    assert sg.data.illuminated.any()


def test_set_positions_rotation():
    """set_positions(t) advances the rotation longitude by q0 + w*t."""
    sg = pr.Spangler(nspangles=50, w=30 * pr.Consts.deg, q0=40 * pr.Consts.deg, n_equ=[0, 1, 1])
    sg.populate_spangler(shape="circle", scale=1, seed=1)
    q_before = sg.data.q_equ.iloc[0]
    sg.set_positions(t=1)
    np.testing.assert_allclose(sg.data.q_equ.iloc[0], q_before + 30 * pr.Consts.deg + 40 * pr.Consts.deg)


def test_set_luz_name_filter():
    """set_luz with a name only illuminates the named body."""
    sg1 = pr.Spangler(name="A", nspangles=50)
    sg1.populate_spangler(shape="sphere", scale=1, seed=1)
    sg2 = pr.Spangler(name="B", nspangles=50)
    sg2.populate_spangler(shape="sphere", scale=1, seed=1)
    sgj = pr.Spangler(spanglers=[sg1, sg2])
    sgj.set_positions()
    sgj.set_observer(nvec=[0, 0, 1])
    sgj.set_luz(nvec=[1, 0, 0], name="A")
    assert sgj.data.loc[sgj.data.name == "A", "illuminated"].any()
    assert not sgj.data.loc[sgj.data.name == "B", "illuminated"].any()


# ---------------------------------------------------------------------------
# Placeholder tests for more complex methods.
# These are intentionally left unimplemented (pass) as documentation of what
# still needs coverage. They exercise the full intersection machinery
# (convex hulls, occlusion, shadowing) which is better validated against
# system output than by hand-computed values.
# ---------------------------------------------------------------------------


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


@pytest.mark.skip(reason="Not yet implemented")
def test_update_intersection_state():
    """update_intersection_state computes occlusion via convex hulls."""
    # TODO: verify that update_intersection_state raises AssertionError when no
    # intersection vantage point has been set (empty qhulls), and that it
    # correctly marks hidden_by_int / transit_over_int for an occulting body.
    pass
