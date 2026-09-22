"""Tests for the functions in the ``science`` module."""

import numpy as np
import pytest

import pryngles as pr


def test_spherical_cartesian_roundtrip():
    """``spherical`` and ``cartesian`` are inverses of each other."""
    for xyz in [[1, 1, 1], [-1, 2, 0.5], [0, 0, 1], [3, -4, 2]]:
        rqf = pr.Science.spherical(xyz)
        np.testing.assert_allclose(pr.Science.cartesian(rqf), xyz, atol=1e-12)


def test_spherical_known_value():
    """``spherical`` returns the documented value for [1, 1, 1]."""
    np.testing.assert_allclose(
        pr.Science.spherical([1, 1, 1]),
        [np.sqrt(3), np.pi / 4, 0.61547971],
        atol=1e-6
    )
    # +x axis: radius 1, azimuth 0, elevation 0.
    np.testing.assert_allclose(
        pr.Science.spherical([1, 0, 0]), [1, 0, 0], atol=1e-6
    )
    # +z axis: radius 1, azimuth 0, elevation pi/2.
    np.testing.assert_allclose(
        pr.Science.spherical([0, 0, 1]), [1, 0, np.pi / 2], atol=1e-6
    )


def test_cospherical_known_value():
    """``cospherical`` returns the documented value for [1, 1, 1]."""
    np.testing.assert_allclose(
        pr.Science.cospherical([1, 1, 1]),
        [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(3)],
        atol=1e-6
    )
    # +x axis: cos(az)=1, sin(az)=0, sin(elev)=0.
    np.testing.assert_allclose(
        pr.Science.cospherical([1, 0, 0]), [1, 0, 0], atol=1e-6
    )
    # +z axis: cos(az)=1, sin(az)=0, sin(elev)=1.
    np.testing.assert_allclose(
        pr.Science.cospherical([0, 0, 1]), [1, 0, 1], atol=1e-6
    )


def test_pcylindrical_known_value():
    """``pcylindrical`` returns the documented value for [1, 1, 1]."""
    np.testing.assert_allclose(
        pr.Science.pcylindrical([1, 1, 1]),
        [np.sqrt(2), np.pi / 4, 1 / np.sqrt(3)],
        atol=1e-6
    )
    # +x axis: rho=1, az=0, cos(elev)=0.
    np.testing.assert_allclose(
        pr.Science.pcylindrical([1, 0, 0]), [1, 0, 0], atol=1e-6
    )
    # +z axis: rho=0, az=0, cos(elev)=1.
    np.testing.assert_allclose(
        pr.Science.pcylindrical([0, 0, 1]), [0, 0, 1], atol=1e-6
    )


def test_cartesian_known_value():
    """``cartesian`` returns the documented value for [1, 30deg, 60deg]."""
    np.testing.assert_allclose(
        pr.Science.cartesian([1, 30 * pr.Consts.deg, 60 * pr.Consts.deg]),
        [0.4330127, 0.25, 0.8660254],
        atol=1e-6
    )
    # Azimuth 90 deg, elevation 0 -> +y axis.
    np.testing.assert_allclose(
        pr.Science.cartesian([1, 90 * pr.Consts.deg, 0]), [0, 1, 0], atol=1e-6
    )
    # Elevation 90 deg, azimuth 0 -> +z axis.
    np.testing.assert_allclose(
        pr.Science.cartesian([1, 0, 90 * pr.Consts.deg]), [0, 0, 1], atol=1e-6
    )


def test_direction_roundtrip():
    """``direction`` round-trips between (lon, lat) and cartesian."""
    nvec = pr.Science.direction(120, 45)
    np.testing.assert_allclose(nvec, [-np.sqrt(2)/4, np.sqrt(6)/4, np.sqrt(2)/2], atol=1e-6)
    lon, lat = pr.Science.direction(*nvec)
    np.testing.assert_allclose(lon, 120, atol=1e-6)
    np.testing.assert_allclose(lat, 45, atol=1e-6)


def test_direction_unit_vector():
    """``direction`` returns a unit vector."""
    nvec = pr.Science.direction(30, 60)
    np.testing.assert_allclose(np.linalg.norm(nvec), 1.0, atol=1e-12)


def test_direction_invalid_latitude():
    """``direction`` raises ``ValueError`` for latitude outside [-90, 90]."""
    with pytest.raises(ValueError):
        pr.Science.direction(30, 91)


def test_rotation_matrix_identity():
    """``rotation_matrix`` with ez=[0,0,1] returns the identity."""
    Msys2uni, Muni2sys = pr.Science.rotation_matrix([0, 0, 1], 0)
    np.testing.assert_allclose(Msys2uni, np.identity(3), atol=1e-12)
    np.testing.assert_allclose(Muni2sys, np.identity(3), atol=1e-12)


def test_rotation_matrix_orthonormal():
    """``rotation_matrix`` columns are orthonormal and mutually inverse."""
    Msys2uni, Muni2sys = pr.Science.rotation_matrix([1, 0, -1], 0)
    # Columns are unit vectors.
    np.testing.assert_allclose(np.linalg.norm(Msys2uni, axis=0), 1.0, atol=1e-12)
    # The two matrices are inverses.
    np.testing.assert_allclose(Msys2uni @ Muni2sys, np.identity(3), atol=1e-12)


def test_rotation_matrix_rotates_vector():
    """``rotation_matrix`` rotates a vector and its inverse restores it.

    ``Msys2uni`` rotates a vector from the system frame into the universal
    frame, and ``Muni2sys`` is the inverse rotation, so applying both in
    sequence returns the original vector.
    """
    Msys2uni, Muni2sys = pr.Science.rotation_matrix([1, 0, -1], 0)
    v = np.array([0.3, -0.7, 0.5])
    rotated = Msys2uni @ v
    # The rotation preserves the vector's length.
    np.testing.assert_allclose(np.linalg.norm(rotated), np.linalg.norm(v), atol=1e-12)
    # The rotated vector has the expected values.
    np.testing.assert_allclose(
        rotated, [-np.sqrt(2) / 10, 0.3, -0.6 * np.sqrt(2)], atol=1e-12
    )
    # Applying the inverse rotation restores the original vector.
    np.testing.assert_allclose(Muni2sys @ rotated, v, atol=1e-12)


def test_limb_darkening_linear():
    """``limb_darkening`` is 1 at the center (rho=0) and 1 - c at the edge for a linear model."""
    c = 0.65
    # Check limb darkening in the center of the disk (rho=0).
    np.testing.assert_allclose(pr.Science.limb_darkening(0.0, [c], N=1), 1.0, atol=1e-12)

    # Check limb darkening at the edge of the disk (rho=1).
    np.testing.assert_allclose(pr.Science.limb_darkening(1.0, [c], N=1), 1 - c, atol=1e-12)

    # Check limb darkening at the mid point of the disk radius (rho=0.5).
    mu = np.sqrt(0.75)
    expected = 1 - c * (1 - mu)
    np.testing.assert_allclose(pr.Science.limb_darkening(0.5, [c], N=1), expected, atol=1e-12)


def test_limb_darkening_quadratic():
    """``limb_darkening`` quadratic model matches the analytic form."""
    c0, c1 = 0.6, 0.2

    # Check limb darkening in the center of the disk (rho=0).
    np.testing.assert_allclose(pr.Science.limb_darkening(0.0, [c0, c1], N=1), 1.0, atol=1e-12)

    # Check limb darkening at the edge of the disk (rho=1).
    np.testing.assert_allclose(pr.Science.limb_darkening(1.0, [c0, c1], N=1), 1 - c0 - c1, atol=1e-12)

    # Check limb darkening at the mid point of the disk radius (rho=0.5).
    mu = np.sqrt(0.75)  # mu at rho=0.5
    expected = 1 - c0 * (1 - mu) - c1 * (1 - mu) ** 2
    np.testing.assert_allclose(pr.Science.limb_darkening(0.5, [c0, c1], N=1), expected, atol=1e-12)


def test_limb_darkening_cubic():
    """``limb_darkening`` cubic model matches the analytic form."""
    c0, c1, c2 = 0.9, -0.5, 0.2

    # Check limb darkening in the center of the disk (rho=0).
    np.testing.assert_allclose(pr.Science.limb_darkening(0.0, [c0, c1, c2], N=1), 1.0, atol=1e-12)

    # Check limb darkening at the edge of the disk (rho=1).
    np.testing.assert_allclose(pr.Science.limb_darkening(1.0, [c0, c1, c2], N=1), 1 - c0 - c1 - c2, atol=1e-12)

    # Check limb darkening at the mid point of the disk radius (rho=0.5).
    mu = np.sqrt(0.75)  # mu at rho=0.5
    expected = 1 - c0 * (1 - mu) - c1 * (1 - mu**1.5) - c2 * (1 - mu**2)
    np.testing.assert_allclose(pr.Science.limb_darkening(0.5, [c0, c1, c2], N=1), expected, atol=1e-12)


def test_limb_darkening_quartic():
    """``limb_darkening`` quartic model matches the analytic form."""
    c0, c1, c2, c3 = -0.2, 2.1, -2.0, 0.75

    # Check limb darkening in the center of the disk (rho=0).
    np.testing.assert_allclose(pr.Science.limb_darkening(0.0, [c0, c1, c2, c3], N=1), 1.0, atol=1e-12)

    # Check limb darkening at the edge of the disk (rho=1).
    np.testing.assert_allclose(pr.Science.limb_darkening(1.0, [c0, c1, c2, c3], N=1), 1 - c0 - c1 - c2 - c3, atol=1e-12)

    # Check limb darkening at the mid point of the disk radius (rho=0.5).
    mu = np.sqrt(0.75)  # mu at rho=0.5
    expected = 1 - c0 * (1 - mu**0.5) - c1 * (1 - mu) - c2 * (1 - mu**1.5) - c3 * (1 - mu**2)
    np.testing.assert_allclose(pr.Science.limb_darkening(0.5, [c0, c1, c2, c3], N=1), expected, atol=1e-12)


def test_limb_darkening_order_validation():
    """``limb_darkening`` raises ``ValueError`` for order > 4."""
    with pytest.raises(ValueError):
        pr.Science.limb_darkening(0.5, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_blackbody_intensity_positive():
    """``blackbody_intensity`` is positive for physical inputs."""
    B = pr.Science.blackbody_intensity(500e-9, 5800)
    assert B > 0


def test_blackbody_wien_peak():
    """``blackbody_intensity`` peaks near the Wien displacement law."""
    lam = np.linspace(100e-9, 3000e-9, 1000)
    B = pr.Science.blackbody_intensity(lam, 5800)
    peak = lam[np.argmax(B)]
    wien = 2.898e-3 / 5800
    np.testing.assert_allclose(peak, wien, rtol=0.05)


def test_blackbody_photons_relation():
    """``blackbody_photons`` is pi * B / (hc/lambda)."""
    lam, T = 500e-9, 5800
    B = pr.Science.blackbody_intensity(lam, T)
    J = pr.Science.blackbody_photons(lam, T)
    h, c = 6.62607015e-34, 299792458
    np.testing.assert_allclose(J, np.pi * B / (h * c / lam), rtol=1e-6)


def test_integrate_planck_flux_stefan_boltzmann():
    """``integrate_planck_flux`` over all wavelengths gives sigma*T^4/pi.

    This is the Stefan-Boltzmann law for the specific intensity: integrating
    Planck's law over all wavelengths yields sigma*T^4/pi. The integration
    range (1 nm to 1 mm) captures essentially all of the blackbody flux.
    """
    sigma = 5.670374419e-8
    T = 5800.0
    flux = pr.Science.integrate_planck_flux(T, 1e-9, 1e-3)
    np.testing.assert_allclose(flux, sigma * T**4 / np.pi, rtol=1e-6)


def test_integrate_planck_photons_scales_t3():
    """``integrate_planck_photons`` scales as T^3.

    The total photon number flux from a blackbody is proportional to T^3, so
    doubling the temperature increases the integrated photon flux by a factor
    of 8.
    """
    T = 5800.0
    photons = pr.Science.integrate_planck_photons(T, 1e-9, 1e-3)
    photons_2T = pr.Science.integrate_planck_photons(2 * T, 1e-9, 1e-3)
    np.testing.assert_allclose(photons_2T / photons, 8.0, rtol=1e-4)


def test_get_convexhull():
    """``get_convexhull`` returns a hull for valid data and None for empty."""
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    hull = pr.Science.get_convexhull(pts)
    assert hull is not None
    assert pr.Science.get_convexhull(np.array([])) is None


def test_points_in_hull():
    """``points_in_hull`` correctly identifies points inside/outside."""
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    hull = pr.Science.get_convexhull(pts)
    inside = pr.Science.points_in_hull(
        np.array([[0.5, 0.5], [2, 2], [0.8, 0.1], [0.5, 1.5]]),
        hull
    )
    np.testing.assert_array_equal(inside, [True, False, True, False])


def test_plane_coefficients():
    """``Plane`` computes the documented plane coefficients."""
    p1 = [-1, 2, 1]
    p2 = [0, -3, 2]
    p3 = [1, 1, -4]
    plane = pr.Plane(p1, p2, p3)
    assert plane.a == 26
    assert plane.b == 7
    assert plane.c == 9
    assert plane.d == 3


def test_plane_projection():
    """``get_projection`` returns the documented projection and distance."""
    p1 = [-1, 2, 1]
    p2 = [0, -3, 2]
    p3 = [1, 1, -4]
    plane = pr.Plane(p1, p2, p3)
    p = [2, 2, 5]
    v, d = plane.get_projection(p)
    np.testing.assert_allclose(v, [-1.67741935483871, 1.0099255583126552, 3.727047146401985], atol=1e-6)
    np.testing.assert_allclose(d, 4.015478735955178, atol=1e-6)


def test_plane_is_above_below():
    """``is_above`` and ``is_below`` are complementary."""
    p1 = [-1, 2, 1]
    p2 = [0, -3, 2]
    p3 = [1, 1, -4]
    plane = pr.Plane(p1, p2, p3)
    p = [2, 2, 5]
    assert plane.is_above(p, [0, 0, -1]) is False
    assert plane.is_below(p, [0, 0, -1]) is True


def test_plane_get_z():
    """``get_z`` returns a z that satisfies the plane equation."""
    p1 = [-1, 2, 1]
    p2 = [0, -3, 2]
    p3 = [1, 1, -4]
    plane = pr.Plane(p1, p2, p3)
    # Check get z correctly calculates the original input z values for the three defining points.
    np.testing.assert_allclose(plane.get_z(-1, 2), 1, atol=1e-12)
    np.testing.assert_allclose(plane.get_z(0, -3), 2, atol=1e-12)
    np.testing.assert_allclose(plane.get_z(1, 1), -4, atol=1e-12)

    # Check one extra point
    np.testing.assert_allclose(plane.get_z(0, 0), -1/3, atol=1e-12)
