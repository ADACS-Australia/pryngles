##################################################################
#                                                                #
# .#####...#####...##..##..##..##...####...##......######...####..#
# .##..##..##..##...####...###.##..##......##......##......##.....#
# .#####...#####.....##....##.###..##.###..##......####.....####..#
# .##......##..##....##....##..##..##..##..##......##..........##.#
# .##......##..##....##....##..##...####...######..######...####..#
# ................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################

# --------------------------------------------------
# External required packages
# --------------------------------------------------

import numpy as np
from numba import njit, prange, types
from scipy.interpolate import CubicSpline

from pryngles.common import VERB_SIMPLE, verbose

# Plain numba types
f64 = types.float64
i64 = types.int64

# Read-only C-contiguous float64 numba array types
f64_arr1 = types.Array(f64, 1, "C", readonly=True)
f64_arr4 = types.Array(f64, 4, "C", readonly=True)

# Exclusive upper bound on nmat, i.e. nmat must satisfy 0 < nmat < MAX_NSTOKES.
# The stokes output of reflection() has nmat + 1 columns (the nmat Stokes elements plus
# the degree of polarisation), and its scratch arrays (Bplus, SvR) have MAX_NSTOKES entries.
MAX_NSTOKES = 4

# Maximum deviation (in cos(theta) and cos(theta0)) from the first pixel for all
# pixels to be treated as sharing one angle pair in reflection(). Numba compiles
# this value into reflection(), so reassigning it at runtime has no effect.
SHARED_ANGLE_TOL = 1e-6


def calculate_spline_derivs(xmu, fou):
    """
    Precompute the spline second-derivative tables needed by fourier_at().

    Natural cubic splines are fitted on the xmu grid along each angle axis.

    Parameters
    ----------
    xmu : ndarray, shape (nmugs,)
        Strictly increasing grid of cos(angle) values.
    fou : ndarray, shape (nmugs, nmugs, nfou, nmat)
        Fourier coefficients, with axes (i, j, m, k).

    Returns
    -------
    dict of ndarray
        Each array has the same shape as `fou`, with keys:

        - ``"j"``: d2f/dj2, second derivative along axis 1 (theta0).
        - ``"i"``: d2f/di2, second derivative along axis 0 (theta).
        - ``"ij"``: d2/dj2 of the ``"i"`` table (cross term, d4f/di2dj2).
    """
    # fou has axes (i, j, m, k), so the angle axes i and j are axes 0 and 1

    def d2(y, axis):
        """Second derivative of the natural cubic spline of y along axis, at the xmu nodes."""
        s = CubicSpline(xmu, y, axis=axis, bc_type="natural")
        return np.ascontiguousarray(s(xmu, 2))

    ddf_j = d2(fou, 1)
    ddf_i = d2(fou, 0)

    return {"j": ddf_j, "i": ddf_i, "ij": d2(ddf_i, 1)}


@njit(types.Tuple((i64, f64, f64, f64, f64))(f64_arr1, f64), cache=True)
def spline_weights(x, xi):
    """
    Find the bracketing index and cubic-spline interpolation weights for xi in x.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Grid nodes. Must be strictly increasing with at least 2 points.
    xi : float
        Point at which to interpolate. Values outside the range of `x` use the
        first/last interval.

    Returns
    -------
    i : int
        Index of the node to the left of `xi`, with 0 <= i <= n - 2.
    a, b, c, d : float
        Interpolation weights, such that
        ``f(xi) = a * f[i] + b * f[i+1] + c * d2f[i] + d * d2f[i+1]``
        where ``d2f`` are the spline second derivatives at the nodes.
    """
    # Find the closest index that's less than xi
    # Values outside the range of x are clamped to the first/last interval.
    i = min(max(np.searchsorted(x, xi, side="right") - 1, 0), len(x) - 2)

    h = x[i + 1] - x[i]

    # Compute the corresponding spline weights for xi, given the closest index on the left
    # where the interpolation will be of the form:
    #  f(x) = a * f(x0) + b * f(x1) + c * d2f(x0)/dx2 + d * d2f(x1)/dx2
    a = (x[i + 1] - xi) / h
    b = (xi - x[i]) / h
    c = (a**3 - a) * (h**2 / 6)
    d = (b**3 - b) * (h**2 / 6)

    return i, a, b, c, d


@njit(
    types.Array(f64, 2, "C")(
        f64_arr1,  # xmu
        f64_arr4,  # fou
        f64_arr4,  # ddf_j
        f64_arr4,  # ddf_i
        f64_arr4,  # ddddf_ij
        f64,  # theta0
        f64,  # theta
    ),
    cache=True,
)
def fourier_at(xmu, fou, ddf_j, ddf_i, ddddf_ij, theta0, theta):
    """
    Interpolate the Fourier coefficients at (theta, theta0).

    To get the coefficient at (theta, theta0), interpolate along j at theta0
    for the two rows bracketing theta, then along i at theta. The second pass
    needs d2f/di2 at those rows; it comes from interpolating the precomputed
    ddf_i along j (spline construction along i and interpolation along j
    commute), so no spline is solved per call.

    Parameters
    ----------
    xmu, fou, ddf_j, ddf_i, ddddf_ij : ndarray
        As for `reflection`.
    theta0, theta : float
        cos(theta0) and cos(theta). Values outside the range of `xmu` use the
        first/last grid interval (see `spline_weights`).

    Returns
    -------
    f : ndarray, shape (nfou, nmat)
        Interpolated Fourier coefficients, indexed by (m, k).
    """
    nfou = fou.shape[2]
    nmat = fou.shape[3]
    f = np.empty((nfou, nmat))

    # Grid node and spline weights at theta0 (along j) and theta (along i)
    j, wja, wjb, wjc, wjd = spline_weights(xmu, theta0)
    i, wia, wib, wic, wid = spline_weights(xmu, theta)

    for m in range(nfou):
        for k in range(nmat):
            # Interpolate f and d2f/di2 at theta0, for the two rows (i and i+1) bracketing theta
            f_ileft = (
                wja * fou[i, j, m, k]
                + wjb * fou[i, j + 1, m, k]
                + wjc * ddf_j[i, j, m, k]
                + wjd * ddf_j[i, j + 1, m, k]
            )
            f_iright = (
                wja * fou[i + 1, j, m, k]
                + wjb * fou[i + 1, j + 1, m, k]
                + wjc * ddf_j[i + 1, j, m, k]
                + wjd * ddf_j[i + 1, j + 1, m, k]
            )
            ddf_ileft = (
                wja * ddf_i[i, j, m, k]
                + wjb * ddf_i[i, j + 1, m, k]
                + wjc * ddddf_ij[i, j, m, k]
                + wjd * ddddf_ij[i, j + 1, m, k]
            )
            ddf_iright = (
                wja * ddf_i[i + 1, j, m, k]
                + wjb * ddf_i[i + 1, j + 1, m, k]
                + wjc * ddddf_ij[i + 1, j, m, k]
                + wjd * ddddf_ij[i + 1, j + 1, m, k]
            )

            # Then interpolate along i at theta to get the Fourier coefficient (m, k)
            f[m, k] = wia * f_ileft + wib * f_iright + wic * ddf_ileft + wid * ddf_iright

    return f


@njit(
    types.Array(f64, 2, "C")(
        f64_arr1,  # xmu
        f64_arr4,  # fou
        f64_arr4,  # ddf_j
        f64_arr4,  # ddf_i
        f64_arr4,  # ddddf_ij
        f64_arr1,  # phi
        f64_arr1,  # beta
        f64_arr1,  # theta0
        f64_arr1,  # theta
        f64_arr1,  # apix
    ),
    cache=True,
    parallel=True,
)
def reflection(xmu, fou, ddf_j, ddf_i, ddddf_ij, phi, beta, theta0, theta, apix):
    """
    Calculate the Stokes vector of each pixel from a table of Fourier coefficients.

    It is assumed that the Stokes vector of the incoming sunlight is
    [1, 0, 0, 0], with the flux measured perpendicular to the direction of
    incidence equal to pi.

    Parameters
    ----------
    xmu : ndarray, shape (nmugs,)
        Strictly increasing grid of cos(angle) values.
    fou : ndarray, shape (nmugs, nmugs, nfou, nmat)
        Fourier coefficients, read from a file.
    ddf_j, ddf_i, ddddf_ij : ndarray, shape (nmugs, nmugs, nfou, nmat)
        Precomputed spline second-derivative tables for `fou` on the `xmu` grid.
    phi, beta : ndarray, shape (npix,)
        Angles in radians.
    theta0, theta : ndarray, shape (npix,)
        cos(theta0) and cos(theta) respectively.
    apix : ndarray, shape (npix,)
        Pixel areas.

    Returns
    -------
    stokes : ndarray, shape (npix, nmat + 1)
        Row n holds the nmat Stokes elements of pixel n followed by the degree
        of polarisation.

    Notes
    -----
    Pixels are independent, so the loop over them runs in parallel (prange).
    If all pixels share the same (theta, theta0), to within `SHARED_ANGLE_TOL`,
    the interpolated Fourier coefficients are computed only once (at the angles
    of the first pixel) and reused for every pixel.

    Inputs are not validated; see the assumptions listed at the top of the
    function body.
    """
    nfou = fou.shape[2]
    nmat = fou.shape[3]
    npix = len(phi)

    # Note, this function assumes (and does not check):
    #   - xmu is strictly increasing, with at least 2 points
    #   - fou, ddf_j, ddf_i and ddddf_ij all have the same shape (nmugs, nmugs, nfou, nmat)
    #   - phi, beta, theta0, theta and apix all have the same length (npix)
    #   - nfou >= 1, npix >= 1, and 0 < nmat < MAX_NSTOKES

    stokes = np.zeros((npix, nmat + 1))

    # The Fourier coefficients depend only on (theta, theta0), not on phi or
    # beta, so if every pixel has the same angle pair they can be interpolated
    # once here and reused below.
    #
    # Every pixel is compared against pixel 0 (not against its neighbour), so the
    # deviation from the shared pair is bounded by SHARED_ANGLE_TOL and cannot
    # drift along the array. The test is written as "not (|diff| <= tol)" rather
    # than "|diff| > tol" on purpose: any comparison with NaN is False, so a NaN
    # angle fails the "<=" test and correctly disables sharing, whereas
    # "|diff| > tol" would be False for NaN and wrongly keep it.
    shared = True
    for n in range(1, npix):
        if not (abs(theta[n] - theta[0]) <= SHARED_ANGLE_TOL and abs(theta0[n] - theta0[0]) <= SHARED_ANGLE_TOL):
            shared = False
            break

    # If every pixel shares one angle pair, interpolate once here (at pixel 0's
    # angles) and reuse the result for all pixels. Otherwise f_shared is never
    # assigned or read, and each pixel interpolates for itself in the loop below.
    if shared:
        f_shared = fourier_at(xmu, fou, ddf_j, ddf_i, ddddf_ij, theta0[0], theta[0])

    for n in prange(npix):
        # Per-pixel scratch arrays. They are allocated inside the loop body so
        # that each parallel iteration has its own copy (no shared state).
        RM = np.zeros(nmat)
        Bplus = np.zeros(MAX_NSTOKES)
        SvR = np.zeros(MAX_NSTOKES)

        # Fourier coefficients (m, k) at this pixel's angles
        if shared:
            f = f_shared
        else:
            f = fourier_at(xmu, fou, ddf_j, ddf_i, ddddf_ij, theta0[n], theta[n])

        # Loop over the Fourier coefficients, summing their contributions
        # to the reflection matrix of pixel n.
        for m in range(nfou):
            fac = 0.5 if m == 0 else 1.0
            cosmphi = np.cos(m * phi[n])
            sinmphi = np.sin(m * phi[n])
            Bplus[0] = cosmphi
            Bplus[1] = cosmphi
            Bplus[2] = sinmphi
            Bplus[3] = sinmphi

            for k in range(nmat):
                # Compute the reflection matrix at pixel n and stokes parameter k,
                # summing over fourier terms m
                RM[k] += 2 * Bplus[k] * fac * f[m, k]

        # Calculate the locally reflected Stokes vector
        for k in range(nmat):
            SvR[k] = theta0[n] * RM[k]

        # Rotate Stokes elements Q and U to the actual reference plane.
        # This does not assume nmat=3: SvR has MAX_NSTOKES entries, zero beyond
        # nmat, so SvR[1] and SvR[2] are always valid and any missing element
        # is treated as zero.
        cosb2 = np.cos(2 * beta[n])
        sinb2 = np.sin(2 * beta[n])
        SvR2 = cosb2 * SvR[1] + sinb2 * SvR[2]
        SvR3 = -sinb2 * SvR[1] + cosb2 * SvR[2]
        SvR[1] = SvR2
        SvR[2] = SvR3

        # Compute the local degree of polarisation P:
        if abs(SvR[0]) < 1e-6:
            P = 0.0
        elif abs(SvR[2]) < 1e-6:
            P = -SvR[1] / SvR[0]
        else:
            P = np.sqrt(SvR[1] * SvR[1] + SvR[2] * SvR[2]) / SvR[0]

        if abs(P) < 1e-6:
            P = 0.0

        # Multiply with theta[n] and the actual pixel area to obtain the stokes
        # element k for each pixel n.
        for k in range(nmat):
            stokes[n, k] = SvR[k] * theta[n] * apix[n]

        # The value of the degree of polarization (last index)
        stokes[n, nmat] = P

    return stokes


class StokesScatterer:
    """Stokes scatterer"""

    def __init__(self, filename):
        """
        Read a file containing Fourier coefficients produced by PyMieDAP.

        Parameters
        ----------
        filename : str
            Path of the file to read.

        Raises
        ------
        RuntimeError
            If the file is not dense (one row per (m, i, j)), or if nfou or nmat
            are outside the range supported by `reflection`.

        Attributes
        ----------
        nmugs : int
            Number of Gaussian integration coefficients.
        nmat : int
            Number of matrix elements.
        nfou : int
            Number of Fourier coefficients.
        xmu : ndarray, shape (nmugs,)
            Grid of cos(angle) values.
        rfou : ndarray, shape (nmugs, nmugs, nfou, nmat)
            Fourier coefficients for reflection, with axes (i, j, m, k).
        rtra : ndarray, shape (nmugs, nmugs, nfou, nmat)
            Fourier coefficients for transmission, with axes (i, j, m, k). Zeros if
            the file has no transmission data.
        rfou_derivs, rtra_derivs : dict of ndarray
            Spline second-derivative tables for `rfou` and `rtra`, as returned by
            `calculate_spline_derivs`.
        """
        self.filename = filename

        # Drop comments and blank lines
        with open(filename) as f:
            lines = [line for line in f if "#" not in line and line.strip()]

        # Read header:
        # First two lines are nmat and nmugs
        nmat, nmugs = int(lines[0]), int(lines[1])

        # then the next nmugs lines are the xmu values,
        xmu = np.array([float(line.split()[0]) for line in lines[2 : 2 + nmugs]])

        # Rest is the core data: m i j rfou... rtra...
        data = np.loadtxt(lines[2 + nmugs :])
        nfou = int(data[:, 0].max()) + 1

        # Ensure the size is as expected
        if len(data) != nfou * nmugs * nmugs:
            raise RuntimeError(
                f"Expected a dense file with one row per (m, i, j), "
                f"but got {len(data)} rows instead of {nfou * nmugs * nmugs}"
            )

        # Check the sizes are supported by reflection()
        if nfou < 1 or not (0 < nmat < MAX_NSTOKES):
            raise RuntimeError(
                f"Invalid array sizes in '{filename}': nfou={nfou}, nmat={nmat} "
                f"(need nfou >= 1 and 0 < nmat < {MAX_NSTOKES})"
            )

        # Index columns
        m, i, j = (data[:, c].astype(int) for c in range(3))

        # Function which defines how to store the fourier coefficients, where
        #   i,j: angle indices
        #     m: fourier coefficient index
        #     k: matrix element index
        # The angle axes come first so that, for a given pair of angles, the
        # coefficients for all (m, k) are contiguous in memory.
        def fill(array):
            out = np.zeros((nmugs, nmugs, nfou, nmat))  # axes: i, j, m, k
            out[i - 1, j - 1, m, :] = array
            return out

        # Get reflection coefficients
        rfou = fill(data[:, 3 : 3 + nmat])
        rfou_derivs = calculate_spline_derivs(xmu, rfou)

        # Get transmission coefficients if they exist, otherwise fill with zeros
        if data.shape[1] > 3 + nmat:
            rtra = fill(data[:, 3 + nmat : 3 + 2 * nmat])
            rtra_derivs = calculate_spline_derivs(xmu, rtra)
        else:
            rtra = np.zeros_like(rfou)
            rtra_derivs = {k: np.zeros_like(rfou) for k in ("i", "j", "ij")}

        verbose(VERB_SIMPLE, f"Checksum '{filename}': {rfou.sum() + rtra.sum():.16e}")

        self.nmat, self.nmugs, self.nfou = nmat, nmugs, nfou
        self.xmu, self.rfou, self.rtra = xmu, rfou, rtra
        self.rfou_derivs, self.rtra_derivs = rfou_derivs, rtra_derivs

    def calculate_stokes(self, phi, beta, theta0, theta, apix, qreflection=1):
        """
        Calculate the Stokes vector of each pixel.

        Uses the reflection or transmission Fourier coefficients.

        Parameters
        ----------
        phi, beta, theta0, theta, apix : ndarray, shape (npix,)
            C-contiguous float64 arrays, all of the same length. `phi` and `beta`
            are in radians; `theta0` and `theta` are cos(theta0) and cos(theta);
            `apix` is the pixel area.
        qreflection : bool, optional
            If true (default) use the reflection coefficients, otherwise the
            transmission ones.

        Returns
        -------
        stokes : ndarray, shape (npix, nmat + 1)
            Row n holds the nmat Stokes elements of pixel n followed by the degree
            of polarisation.

        Raises
        ------
        RuntimeError
            If the input arrays do not all have the same length.
        """
        npix = len(phi)
        stokes = np.zeros((npix, self.nmat + 1))

        if len(beta) != npix or len(theta0) != npix or len(theta) != npix or len(apix) != npix:
            raise RuntimeError("Input arrays must have the same length")

        # Return early if there are no pixels to process (i.e. zero-length arrays)
        if npix == 0:
            return stokes

        if qreflection:
            fourier_coefficients = self.rfou
            fourier_derivs = self.rfou_derivs
        else:
            fourier_coefficients = self.rtra
            fourier_derivs = self.rtra_derivs

        stokes = reflection(
            self.xmu,
            fourier_coefficients,
            fourier_derivs["j"],
            fourier_derivs["i"],
            fourier_derivs["ij"],
            phi,
            beta,
            theta0,
            theta,
            apix,
        )

        return stokes
