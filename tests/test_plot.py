"""Tests for the functions in the ``plot`` module."""

import matplotlib
matplotlib.use("Agg")  # headless backend so tests don't need a display

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import art3d

import pryngles as pr


def test_rgb_primary_colors():
    """``rgb`` converts HLS values to the expected RGB for primary hues."""
    # Hue 0 -> red, hue 120 -> green, hue 240 -> blue (full saturation, mid level)
    np.testing.assert_allclose(pr.Plot.rgb([0, 0.5, 1]), (1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(pr.Plot.rgb([120, 0.5, 1]), (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(pr.Plot.rgb([240, 0.5, 1]), (0.0, 0.0, 1.0), atol=1e-6)


def test_rgb_extremes():
    """``rgb`` returns black at level 0 and white at level 1."""
    np.testing.assert_allclose(pr.Plot.rgb([0, 0, 1]), (0.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(pr.Plot.rgb([0, 1, 1]), (1.0, 1.0, 1.0), atol=1e-6)


def test_rgb_to_hex():
    """``rgb`` with ``to_hex=True`` returns a valid hex color string."""
    hex_red = pr.Plot.rgb([0, 0.5, 1], to_hex=True)
    hex_green = pr.Plot.rgb([120, 0.5, 1], to_hex=True)
    hex_blue = pr.Plot.rgb([240, 0.5, 1], to_hex=True)
    assert isinstance(hex_red, str)
    assert hex_red.startswith("#")
    assert len(hex_red) == 7
    # Red hue at full saturation should be a pure red hex.
    assert hex_red == "#ff0000"
    assert hex_green == "#00ff00"
    assert hex_blue == "#0000ff"


def test_calc_flyby_shape():
    """``calc_flyby`` returns one unit vector per requested point."""
    nvecs = pr.Plot.calc_flyby(normal=[0, 0, 1], lat=0, num=10)
    assert nvecs.shape == (10, 3)
    # Each returned vector is a unit vector.
    norms = np.linalg.norm(nvecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_calc_flyby_num_controls_points():
    """``num`` controls the number of returned points."""
    nvecs = pr.Plot.calc_flyby(normal=[0, 0, 1], lat=0, num=5)
    assert nvecs.shape == (5, 3)


def test_calc_flyby_equatorial_plane():
    """With ``normal=[0,0,1]`` and ``lat=0`` the flyby follows the unit circle."""
    # For normal=[0,0,1] the rotation matrix is identity, so each point is
    # simply the unit direction (cos(lon), sin(lon), 0).
    num_points = 10
    nvecs = pr.Plot.calc_flyby(normal=[0, 0, 1], lat=0, num=num_points, start=0, stop=360)

    for i, vec in enumerate(nvecs):
        lon = i * (360 / (num_points - 1))  # evenly spaced intervals
        expected = (np.cos(np.radians(lon)), np.sin(np.radians(lon)), 0.0)
        np.testing.assert_allclose(vec, expected, atol=1e-6)


def test_calc_flyby_latitude():
    """A non-zero latitude raises the flyby out of the equatorial plane."""
    lat = 30.0
    num_points = 10
    nvecs = pr.Plot.calc_flyby(normal=[0, 0, 1], lat=lat, num=num_points, start=0, stop=360)

    # ``Science.direction`` uses ``spy.latrec``, so the x/y components are
    # scaled by ``cos(lat)`` while the z component is ``sin(lat)``.
    for i, vec in enumerate(nvecs):
        lon = i * (360 / (num_points - 1))  # evenly spaced intervals
        expected = (
            np.cos(np.radians(lat)) * np.cos(np.radians(lon)),
            np.cos(np.radians(lat)) * np.sin(np.radians(lon)),
            np.sin(np.radians(lat)),
        )
        np.testing.assert_allclose(vec, expected, atol=1e-6)

    # Vectors remain unit vectors.
    norms = np.linalg.norm(nvecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_calc_flyby_start_stop():
    """``start`` and ``stop`` control the longitude range of the flyby."""
    # A flyby from 0 to 90 degrees should start at (1,0,0) and end at (0,1,0).
    nvecs = pr.Plot.calc_flyby(normal=[0, 0, 1], lat=0, num=3, start=0, stop=90)
    np.testing.assert_allclose(nvecs[0], (1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(nvecs[-1], (0.0, 1.0, 0.0), atol=1e-6)


def test_pryngles_mark_returns_text():
    """``pryngles_mark`` adds a watermark and returns a matplotlib ``Text``."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    text = pr.Plot.pryngles_mark(ax)
    assert isinstance(text, plt.Text)
    # The watermark text should carry the pryngles version and the configured
    # styling (rotation, alignment, color, zorder).
    assert text.get_text() == f"Pryngles {pr.version}"
    assert text.get_rotation() == 270
    assert text.get_ha() == "left"
    assert text.get_va() == "top"
    assert text.get_color() == "pink"
    assert text.get_zorder() == 100
    plt.close(fig)


def test_circle3d_adds_patch():
    """``circle3d`` adds a patch to a 3d axes without raising."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    n_patches_before = len(ax.patches)
    patch_radius = 0.5
    patch_alpha = 0.3
    pr.Plot.circle3d(ax, (0, 0, 0), patch_radius, zDir=[1, 1, 0], fill="None", alpha=patch_alpha)
    assert len(ax.patches) == n_patches_before + 1
    # The added patch is converted to a 3d path patch, but retains the original
    # 2d circle geometry in ``_path2d`` (before the 3d rotation is applied).
    patch = ax.patches[-1]
    assert isinstance(patch, art3d.PathPatch3D)
    # No easy way to check the radius of the created patch without checking the bounding box.
    bounds = patch._path2d.get_extents()
    np.testing.assert_allclose(bounds.x0, -patch_radius, atol=1e-6)
    np.testing.assert_allclose(bounds.x1, patch_radius, atol=1e-6)
    np.testing.assert_allclose(bounds.y0, -patch_radius, atol=1e-6)
    np.testing.assert_allclose(bounds.y1, patch_radius, atol=1e-6)
    assert patch.get_alpha() == patch_alpha
    plt.close(fig)


def test_rgb_sample_runs():
    """``rgb_sample`` creates a figure without raising."""
    pr.Plot.rgb_sample(H=0)
    assert plt.get_fignums()
    plt.close("all")


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")
def test_animate_rebound_returns_animation():
    """``animate_rebound`` returns a matplotlib animation for a rebound sim."""
    sim = pr.rb.Simulation()
    ms = 1
    sim.add(m=ms)
    mp = 1e-3
    xp = 0.3
    vp = np.sqrt(sim.G * ms / xp)
    sim.add(m=mp, x=xp, vy=vp)

    # ``traces=False`` is required to get an animation back (with ``traces=True``
    # the function only draws static traces and returns ``None``).
    anim = pr.Plot.animate_rebound(sim, traces=False, nsnap=5, axis=True, ms=1)
    assert isinstance(anim, matplotlib.animation.ArtistAnimation)
    plt.close("all")
