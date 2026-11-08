import pickle

import numpy as np
import pandas as pd
import pytest

import pryngles as pr
from pryngles import science


def test_const():
    """SPANGLER_KEY_ORDERING and SPANGLER_COLUMNS must contain the same keys."""
    for key in pr.consts.SPANGLER_KEY_ORDERING:
        assert key in pr.consts.SPANGLER_COLUMNS, f"Column '{key}' in SPANGLER_KEY_ORDERING not in SPANGLER_COLUMNS"
    for key in pr.consts.SPANGLER_COLUMNS:
        assert key in pr.consts.SPANGLER_KEY_ORDERING, (
            f"Column '{key}' in SPANGLER_COLUMNS not in SPANGLER_KEY_ORDERING"
        )


def test_init_basic():
    """A basic Spangler has the expected number of rows and default state."""
    sg = pr.Spangler(nspangles=3, center_equ=[0, 0, 0], n_equ=[1, 0, 0])
    assert sg.nspangles == 3
    assert len(sg.data) == 3
    assert sg.shape == "vanilla"
    # Default state: unset True, visibility/source states False
    assert sg.data.unset.all()
    for col in list(pr.consts.SPANGLER_VISIBILITY_STATES) + list(pr.consts.SPANGLER_SOURCE_STATES):
        assert not sg.data[col].any()


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
    assert sg.data.unset.all()
    for col in list(pr.consts.SPANGLER_VISIBILITY_STATES) + list(pr.consts.SPANGLER_SOURCE_STATES):
        assert not sg.data[col].any()
    for coords in "int", "obs", "luz":
        assert (sg.data["hidden_mask_" + coords] == 0).all()
        assert (sg.data["transit_mask_" + coords] == 0).all()
        assert sg.data["transit_rho_" + coords].isna().all()


def test_get_set_cols_vector():
    """_set_cols writes masked rows of several columns; _get_cols reads them back."""
    sg = pr.Spangler(nspangles=10, center_equ=[0, 0, 0], n_equ=[1, 0, 0])
    cols = pr.consts.SPANGLER_VEC_GROUPS["center_int"]
    mask = np.arange(sg.nspangles) % 2 == 0
    before = sg.data.copy()

    values = np.arange(15, dtype=float).reshape(5, 3)
    sg._set_cols(cols, values, mask)

    np.testing.assert_array_equal(sg._get_cols(cols, mask), values)
    np.testing.assert_array_equal(sg.data.loc[~mask, cols].to_numpy(), before.loc[~mask, cols].to_numpy())
    assert (sg.data.dtypes == before.dtypes).all()


def test_set_cols_broadcast():
    """A 1D value is one value per column; a scalar fills every selected row."""
    sg = pr.Spangler(nspangles=10, center_equ=[0, 0, 0], n_equ=[1, 0, 0])
    cols = pr.consts.SPANGLER_VEC_GROUPS["center_int"]
    mask = np.arange(sg.nspangles) < 4

    sg._set_cols(cols, [1.0, 2.0, 3.0], mask)
    np.testing.assert_array_equal(sg._get_cols(cols, mask), np.tile([1.0, 2.0, 3.0], (4, 1)))

    sg._set_cols("d_int", np.inf, mask)
    assert np.isinf(sg._get_cols("d_int", mask)).all()


def test_set_cols_keeps_dtypes():
    """Bool, string and int columns keep their dtype after a masked write."""
    sg = pr.Spangler(nspangles=10, center_equ=[0, 0, 0], n_equ=[1, 0, 0])
    mask = np.arange(sg.nspangles) < 3
    dtypes = sg.data.dtypes.copy()

    sg._set_cols("illuminated", True, mask)
    sg._set_cols("emmitter", "lambert", mask)
    sg._set_cols("geometry", 7, mask)

    assert (sg.data.dtypes == dtypes).all()
    assert sg.data.illuminated.tolist() == [True] * 3 + [False] * 7
    assert (sg.data.loc[mask, "emmitter"] == "lambert").all()
    assert (sg.data.loc[mask, "geometry"] == 7).all()


def test_set_cols_rejects_lossy_cast():
    """Writing floats into an int column raises instead of truncating."""
    sg = pr.Spangler(nspangles=10, center_equ=[0, 0, 0], n_equ=[1, 0, 0])
    with pytest.raises(TypeError):
        sg._set_cols("geometry", 1.5)


def test_set_scale():
    """set_scale scales lengths by scale, areas by scale**2, vectors by scale."""
    sg = pr.Spangler(nspangles=10, center_equ=[1, 2, 3])
    sg.populate_spangler(shape="circle", scale=1, seed=1)
    sg.set_positions()

    asp_before = sg.data.asp.iloc[0]
    x_before = sg.data.x_equ.iloc[0]
    center_before = sg.data.vectors.center_equ.iloc[0].to_numpy()

    scale = 3
    sg.set_scale(scale)

    assert sg.scale == scale
    np.testing.assert_allclose(sg.data.asp.iloc[0], asp_before * scale**2)
    np.testing.assert_allclose(sg.data.x_equ.iloc[0], x_before * scale)
    np.testing.assert_allclose(sg.data.vectors.center_equ.iloc[0].to_numpy(), center_before * scale)


def test_populate_spangler_sphere():
    """Sphere spangles lie on a sphere of radius scale with unit normals."""
    scale = 2
    sg = pr.Spangler(nspangles=100)
    sg.populate_spangler(shape="sphere", scale=scale, seed=1)
    sg.set_positions()
    r = np.linalg.norm(sg.data[["x_equ", "y_equ", "z_equ"]].values, axis=1)
    np.testing.assert_allclose(r, scale, atol=1e-6)
    ns = sg.data.vectors.ns_equ.to_numpy()
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
    np.testing.assert_allclose(sg.rqf_obs, science.spherical([0, 0, 1]))
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
    np.testing.assert_allclose(sg.rqf_luz, science.spherical([1, 0, 0]))
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
# Storage: the spangle data is kept as one numpy array per column, and
# Spangler.data builds a pandas DataFrame on every access.
# ---------------------------------------------------------------------------


def _sphere_spangler(name="A", nspangles=20, seed=1):
    sg = pr.Spangler(name=name, nspangles=nspangles)
    sg.populate_spangler(shape="sphere", scale=1, seed=seed)
    sg.set_positions()
    return sg


def test_data_is_a_snapshot():
    """Modifying the DataFrame returned by .data does not change the spangler."""
    sg = _sphere_spangler()
    data = sg.data
    data["Tem"] = 1234.0
    data.loc[0, "visible"] = True
    assert (sg.data["Tem"] != 1234.0).all()
    assert not sg.data.loc[0, "visible"]
    assert sg.data is not sg.data


def test_data_setter_replaces_the_storage():
    """Assigning a whole DataFrame replaces the data, keeping its dtypes."""
    sg = _sphere_spangler()
    data = sg.data
    data["Tem"] = 1234.0
    sg.data = data
    np.testing.assert_array_equal(sg._get_cols("Tem"), np.full(sg.nspangles, 1234.0))
    assert (sg.data.dtypes == data.dtypes).all()


def test_data_round_trip_is_identical():
    """Rebuilding the DataFrame from the arrays gives back the same table (values, dtypes, index)."""
    sg = _sphere_spangler()
    before = sg.data
    sg.data = before
    pd.testing.assert_frame_equal(sg.data, before, check_exact=True)
    assert isinstance(sg.data.index, pd.RangeIndex)


def test_get_cols_without_mask_is_read_only():
    """Unmasked reads are views of the stored arrays and cannot be written to; masked reads are copies."""
    sg = _sphere_spangler()
    with pytest.raises(ValueError):
        sg._get_cols("Tem")[0] = 0.0
    masked = sg._get_cols("Tem", np.ones(sg.nspangles, dtype=bool))
    masked[0] = 0.0
    assert sg._get_cols("Tem")[0] != 0.0


def test_add_col():
    """_add_col adds a filled column with the dtype pandas would infer; existing names are rejected."""
    sg = _sphere_spangler()
    sg._add_col("extra", 0.5)
    assert sg._columns()[-1] == "extra"
    np.testing.assert_array_equal(sg._get_cols("extra"), np.full(sg.nspangles, 0.5))
    assert sg.data["extra"].dtype == np.float64
    with pytest.raises(ValueError):
        sg._add_col("extra", 1.0)


def test_astype():
    """_astype converts one column with pandas' rules and records the new dtype."""
    sg = _sphere_spangler()
    sg._astype("scatterer", object)
    assert sg._column_dtypes()["scatterer"] == np.dtype(object)
    assert sg.data["scatterer"].dtype == np.dtype(object)
    assert (sg.data["scatterer"] == "").all()


def test_join_spanglers_matches_pandas_concat():
    """Joining spanglers array by array gives the same table as pandas.concat of their data."""
    sg1 = _sphere_spangler("A", seed=1)
    sg2 = _sphere_spangler("B", seed=2)
    expected = pd.concat([sg1.data, sg2.data], ignore_index=True)
    sgj = pr.Spangler(spanglers=[sg1, sg2])
    pd.testing.assert_frame_equal(sgj.data, expected, check_exact=True)
    assert sgj.nspangles == sg1.nspangles + sg2.nspangles


def test_join_spanglers_with_different_dtypes_uses_pandas_concat():
    """When column dtypes differ between spanglers, the join falls back to pandas.concat (type promotion)."""
    sg1 = _sphere_spangler("A", seed=1)
    sg2 = _sphere_spangler("B", seed=2)
    sg2._astype("scatterer", object)
    expected = pd.concat([sg1.data, sg2.data], ignore_index=True)
    sgj = pr.Spangler(spanglers=[sg1, sg2])
    pd.testing.assert_frame_equal(sgj.data, expected, check_exact=True)
    assert sgj.data["scatterer"].dtype == np.dtype(object)


def test_unpickle_spangler_saved_with_a_dataframe():
    """Spanglers pickled before the numpy storage (table in a "data" DataFrame) are converted on load."""
    sg = _sphere_spangler()
    expected = sg.data
    old_state = {k: v for k, v in sg.__dict__.items() if k not in ("_cols", "_dtypes")}
    old_state["data"] = expected
    loaded = pr.Spangler.__new__(pr.Spangler)
    loaded.__setstate__(old_state)
    assert "data" not in loaded.__dict__
    pd.testing.assert_frame_equal(loaded.data, expected, check_exact=True)
    loaded._set_cols("visible", True)
    assert loaded.data.visible.all()


def test_pickle_round_trip():
    """A spangler survives pickling with the numpy storage."""
    sg = _sphere_spangler()
    loaded = pickle.loads(pickle.dumps(sg))
    pd.testing.assert_frame_equal(loaded.data, sg.data, check_exact=True)


def test_body_bit():
    """body_bit gives one bit per body, in the order of the spangler's names."""
    sgj = pr.Spangler(spanglers=[_sphere_spangler("A"), _sphere_spangler("B"), _sphere_spangler("C")])
    assert [sgj.body_bit(n) for n in ("A", "B", "C")] == [1, 2, 4]
    with pytest.raises(ValueError):
        sgj.body_bit("D")


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
    # correctly marks hidden_mask_int / transit_mask_int for an occulting body.
    pass
