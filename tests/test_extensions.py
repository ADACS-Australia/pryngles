"""Tests for the functions in the ``extensions`` module."""

import numpy as np
import pryngles as pr


def test_stokes_planet():
    """``calculate_stokes`` matches the reference planet values."""
    S = pr.StokesScatterer(pr.Misc.get_data("fou_gasplanet.dat"))
    stokes = S.calculate_stokes(
        np.array([1.0448451569439827]),
        np.array([3.069394277348945]),
        np.array([0.04990329026929557]),
        np.array([0.02509670973070432]),
        np.array([9.432328787795567e-05]),
    )
    expected = np.array([
        4.597857424902560283e-07,
        2.251229972872198058e-07,
        1.834400800563127439e-09,
        4.896421313424954569e-01
    ])
    np.testing.assert_allclose(stokes[0], expected, rtol=1e-10)


def test_stokes_ring_backscattering():
    """``calculate_stokes`` matches the reference ring backscattering values."""
    S = pr.StokesScatterer(pr.Misc.get_data("fou_ring_0_4_0_8.dat"))
    stokes = S.calculate_stokes(
        np.array([1.490116119384765625e-08]),
        np.array([0.0]),
        np.array([4.999999999999998890e-01]),
        np.array([5.000000000000000000e-01]),
        np.array([1.163314390931110409e-04]),
    )
    expected = np.array([
        6.188864122152382e-06,
        -3.048241433537231e-07,
        -9.751836359244912e-15,
        4.9253649351039624e-2,
    ])
    np.testing.assert_allclose(stokes[0], expected, rtol=1e-10)


def test_stokes_ring_forwardscattering():
    """``calculate_stokes`` matches the reference ring forwardscattering values."""
    S = pr.StokesScatterer(pr.Misc.get_data("fou_ring_0_4_0_8.dat"))
    stokes = S.calculate_stokes(
        np.array([1.601029385538801364e+00]),
        np.array([1.601029385538801364e+00]),
        np.array([1.744974835125044643e-02]),
        np.array([5.000000000000000000e-01]),
        np.array([1.163314390931110409e-04]),
        qreflection=0,
    )
    expected = np.array([
        1.6764574581874426e-07,
        -1.482737896134635e-08,
        2.6735487288654863e-09,
        8.987097703039343e-02,
    ])
    np.testing.assert_allclose(stokes[0], expected, rtol=1e-10)


def test_read_fourier(tmp_path):
    """``read_fourier`` parses the header and core data of a fourier file."""
    # A minimal synthetic file: nmat=3, nmugs=2, then the two xmu rows,
    # then the core data rows (m, i, j, rfou..., rtra...).
    content = (
        "# comment line\n"
        "3\n"
        "2\n"
        "0.1 0.2\n"
        "0.3 0.4\n"
        "0 1 1 1.0 2.0 3.0 4.0 5.0 6.0\n"
        "0 1 2 7.0 8.0 9.0 10.0 11.0 12.0\n"
    )
    path = tmp_path / "fou_test.dat"
    path.write_text(content)

    S = pr.StokesScatterer(str(path))

    assert S.nmat == 3
    assert S.nmugs == 2
    np.testing.assert_allclose(S.xmu, [0.1, 0.3])
    assert S.rfou.shape == (3 * 2, 2, 1)
    assert S.rtra.shape == (3 * 2, 2, 1)
    # First data row: m=0, i=1, j=1 -> rfou[0:3, 0, 0] = [1,2,3],
    # rtra[0:3, 0, 0] = [4,5,6].
    np.testing.assert_allclose(S.rfou[0:3, 0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(S.rtra[0:3, 0, 0], [4.0, 5.0, 6.0])
    # Second data row: m=0, i=1, j=2 -> rfou[0:3, 1, 0] = [7,8,9],
    # rtra[0:3, 1, 0] = [10,11,12].
    np.testing.assert_allclose(S.rfou[0:3, 1, 0], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(S.rtra[0:3, 1, 0], [10.0, 11.0, 12.0])
