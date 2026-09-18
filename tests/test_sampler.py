"""Tests for the functions in the ``sampler`` module."""

import numpy as np
import pytest

import pryngles as pr


def test_gen_circle_geometry():
    """``gen_circle`` samples points in the xy-plane within the unit circle."""
    S = pr.Sampler(N=100, seed=10)
    S.gen_circle()
    assert S.N == 100
    assert S.dim == 2
    # Points lie in the xy-plane.
    np.testing.assert_allclose(S.ss[:, 2], 0.0, atol=1e-12)
    # Points stay within the unit circle.
    radii = np.linalg.norm(S.ss[:, :2], axis=1)
    assert radii.max() <= 1.0 + 1e-12
    # Total area is that of the unit circle.
    np.testing.assert_allclose(S.A, np.pi, atol=1e-12)


def test_gen_circle_seed_reproducible():
    """A fixed seed produces identical samples."""
    S1 = pr.Sampler(N=100, seed=10)
    S1.gen_circle()
    S2 = pr.Sampler(N=100, seed=10)
    S2.gen_circle()
    np.testing.assert_array_equal(S1.ss, S2.ss)


def test_gen_sphere_geometry():
    """``gen_sphere`` samples points on the unit sphere."""
    S = pr.Sampler(N=100, seed=10)
    S.gen_sphere()
    assert S.N == 100
    assert S.dim == 3
    # All points are on the unit sphere.
    norms = np.linalg.norm(S.ss, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)
    # Total area is that of the unit sphere.
    np.testing.assert_allclose(S.A, 4 * np.pi, atol=1e-12)


def test_gen_ring_geometry():
    """``gen_ring`` samples points in an annulus between ``ri`` and 1."""
    ri = 0.6
    S = pr.Sampler(N=500, seed=10)
    S.gen_ring(ri)
    radii = np.linalg.norm(S.ss[:, :2], axis=1)
    # Points stay within the outer radius and outside the inner hole.
    assert radii.max() <= 1.0 + 1e-12
    assert radii.min() >= ri - 1e-12
    # Total area is that of the annulus.
    np.testing.assert_allclose(S.A, np.pi * (1 - ri**2), atol=1e-12)


def test_gen_ring_validation():
    """``gen_ring`` validates the number of points and inner radius."""
    # Must have at least 10 points to sample a ring.
    with pytest.raises(ValueError):
        pr.Sampler(N=5, seed=10).gen_ring(0.5)
    
    # Must have inner radius between 0 and 1.
    with pytest.raises(ValueError):
        pr.Sampler(N=500, seed=10).gen_ring(1.5)


def test_update_normals_circle():
    """``update_normals`` returns +z normals for a circle."""
    S = pr.Sampler(N=100, seed=10)
    S.gen_circle()
    ns = S.update_normals(S.ss)
    np.testing.assert_allclose(ns, np.array([[0, 0, 1]] * S.N), atol=1e-12)


def test_update_normals_sphere():
    """``update_normals`` returns radial (unit) normals for a sphere."""
    S = pr.Sampler(N=100, seed=10)
    S.gen_sphere()
    ns = S.update_normals(S.ss)
    np.testing.assert_allclose(ns, S.ss, atol=1e-12)


def test_calc_distances_consistency():
    """``_calc_distances`` produces consistent derived quantities."""
    S = pr.Sampler(N=100, seed=10)
    S.gen_circle()
    # dmin <= dmed <= dmax.
    assert S.dmin <= S.dmed <= S.dmax
    # dstar = sqrt(N) * dmed.
    np.testing.assert_allclose(S.dstar, np.sqrt(S.N) * S.dmed, atol=1e-12)
    # aes = A / N.
    np.testing.assert_allclose(S.aes, S.A / S.N, atol=1e-12)
    # deff = 2 * sqrt(aes / pi).
    np.testing.assert_allclose(S.deff, 2 * (S.aes / np.pi) ** 0.5, atol=1e-12)


def test_calc_distances_deterministic():
    """``_calc_distances`` computes exact values for a hand-set sample.

    Points along the x-axis at 0, 1, 3, 6, 10 give nearest-neighbour
    distances 1, 1, 2, 3, 4, so every derived quantity is known exactly.
    """
    S = pr.Sampler(N=5, seed=10)
    S.ss = np.array(
        [[0, 0, 0], [1, 0, 0], [3, 0, 0], [6, 0, 0], [10, 0, 0]], dtype=float
    )
    S.N = 5
    S.A = 5.0
    S._calc_distances()

    np.testing.assert_allclose(S.ds, [1.0, 1.0, 2.0, 3.0, 4.0], atol=1e-12)
    assert S.dmin == 1.0
    assert S.dmed == 2.0
    assert S.dmax == 4.0
    assert S.dran == 3.0
    # dstar = sqrt(N) * dmed = sqrt(5) * 2.
    np.testing.assert_allclose(S.dstar, np.sqrt(5) * 2, atol=1e-12)
    # aes = A / N = 5 / 5 = 1.
    np.testing.assert_allclose(S.aes, 1.0, atol=1e-12)
    # deff = 2 * sqrt(aes / pi) = 2 / sqrt(pi).
    np.testing.assert_allclose(S.deff, 2 / np.sqrt(np.pi), atol=1e-12)


def test_cut_hole():
    """``_cut_hole`` removes points inside the inner radius and updates area."""
    S = pr.Sampler(N=1000, seed=10)
    S.gen_circle()
    n_before = S.N
    S._cut_hole(0.5)
    assert S.N < n_before
    radii = np.linalg.norm(S.ss[:, :2], axis=1)
    assert radii.min() >= 0.5 - 1e-12
    np.testing.assert_allclose(S.A, np.pi * (1 - 0.5**2), atol=1e-12)


def test_purge_sample_reduces_n():
    """``purge_sample`` removes close points and marks the sample as purged."""
    S = pr.Sampler(N=1000, seed=10)
    S.gen_sphere()
    n_before = S.N
    S.purge_sample()
    assert S.N <= n_before
    assert S.purged is True


def test_purge_sample_idempotent():
    """``purge_sample`` is a no-op once the sample is already purged."""
    S = pr.Sampler(N=1000, seed=10)
    S.gen_sphere()
    S.purge_sample()
    n_after_first = S.N
    S.purge_sample()
    assert S.N == n_after_first


def test_purge_sample_threshold():
    """A larger ``tol`` purges more aggressively.

    With the same clustered sample, a moderate ``tol`` (0.5) removes one point
    from each cluster, while a large ``tol`` (0.8) removes both points from
    each cluster.
    """
    points = np.array(
        [[0, 0, 0], [0.1, 0, 0], [5, 0, 0], [5.1, 0, 0],
         [10, 0, 0], [20, 0, 0], [30, 0, 0], [40, 0, 0]],
        dtype=float,
    )

    def purge(tol):
        S = pr.Sampler(N=len(points), seed=10)
        S.ss = points.copy()
        S.ns = np.array([[i, 0, 0] for i, _ in enumerate(points)])
        S.pp = np.array([[i, 0, 0] for i, _ in enumerate(points)])
        S.N = len(points)
        S.A = float(len(points))
        S.purged = False
        S.purge_sample(tol=tol)
        return S

    # Moderate tol: one point removed from each cluster.
    S = purge(0.5)
    np.testing.assert_allclose(S.ss[:, 0], [0.1, 5, 10, 20, 30, 40], atol=1e-12)
    np.testing.assert_array_equal(S.ns[:, 0], [1, 2, 4, 5, 6, 7])
    np.testing.assert_array_equal(S.pp[:, 0], [1, 2, 4, 5, 6, 7])
    assert S.N == 6

    # Large tol: both points removed from each cluster.
    S = purge(0.8)
    np.testing.assert_allclose(S.ss[:, 0], [10, 20, 30, 40], atol=1e-12)
    np.testing.assert_array_equal(S.ns[:, 0], [4, 5, 6, 7])
    np.testing.assert_array_equal(S.pp[:, 0], [4, 5, 6, 7])
    assert S.N == 4


def test_preset_circle():
    """A circle preset loads a pre-generated sample from disk."""
    sp = pr.Sampler(preset=("circle", dict()), N=850)
    assert sp.dim == 2
    assert sp.Npreset == 800
    assert sp.N == 800


def test_preset_sphere():
    """A sphere preset loads a pre-generated sample from disk."""
    sp = pr.Sampler(preset=("sphere", dict()), N=2750)
    assert sp.dim == 3
    assert sp.Npreset == 2600
    assert sp.N == 2584


def test_preset_ring():
    """A ring preset loads a circle preset and cuts a hole."""
    sp = pr.Sampler(preset=("ring", dict(ri=0.7)), N=1150)
    assert sp.ri == 0.7
    radii = np.linalg.norm(sp.ss[:, :2], axis=1)
    assert radii.min() >= 0.7 - 1e-12


def test_preset_invalid_geometry():
    """An unknown preset geometry raises ``ValueError``."""
    with pytest.raises(ValueError):
        pr.Sampler(preset=("bogus", dict()), N=100)
