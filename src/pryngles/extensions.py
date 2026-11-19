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

# Maximum number of Stokes elements reflection() supports (size of Bplus etc.)
MAX_NSTOKES = 4

# Maximum deviation (in cos(theta) and cos(theta0)) from the first pixel for all
# pixels to be treated as sharing one angle pair in reflection(). Numba compiles
# this value into reflection(), so reassigning it at runtime has no effect.
SHARED_ANGLE_TOL = 1e-6


def calculate_spline_derivs(xmu, fou):

    def d2(y, axis):
        s = CubicSpline(xmu, y, axis=axis, bc_type="natural")
        return np.ascontiguousarray(s(xmu, 2))

    ddf_j = d2(fou, 3)
    ddf_i = d2(fou, 2)

    return {"j": ddf_j, "i": ddf_i, "ij": d2(ddf_i, 3)}


# Find the bracketing index and interpolation coefficients for xi in x.
# x must be strictly increasing with at least 2 points.
@njit(types.Tuple((i64, f64, f64, f64, f64))(f64_arr1, f64), cache=True)
def spline_weights(x, xi):
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


# ----------------------------------------------------------------------------
#     Return the Fourier coefficients f[m, k] (shape (nfou, nmat)) interpolated
#     at (theta, theta0).
#
#     To get the coefficient at (theta, theta0), interpolate along j at theta0
#     for the two rows bracketing theta, then along i at theta. The second pass
#     needs d2f/di2 at those rows; it comes from interpolating the precomputed
#     ddf_i along j (spline construction along i and interpolation along j
#     commute), so no spline is solved per call.
#
#     The arrays are as for reflection().
# ----------------------------------------------------------------------------
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
    nfou, nmat = fou.shape[:2]
    f = np.empty((nfou, nmat))

    # Grid node and spline weights at theta0 (along j) and theta (along i)
    j, wja, wjb, wjc, wjd = spline_weights(xmu, theta0)
    i, wia, wib, wic, wid = spline_weights(xmu, theta)

    for m in range(nfou):
        for k in range(nmat):
            # Interpolate f and d2f/di2 at theta0, for the two rows (i and i+1) bracketing theta
            f_ileft = (
                wja * fou[m, k, i, j]
                + wjb * fou[m, k, i, j + 1]
                + wjc * ddf_j[m, k, i, j]
                + wjd * ddf_j[m, k, i, j + 1]
            )
            f_iright = (
                wja * fou[m, k, i + 1, j]
                + wjb * fou[m, k, i + 1, j + 1]
                + wjc * ddf_j[m, k, i + 1, j]
                + wjd * ddf_j[m, k, i + 1, j + 1]
            )
            ddf_ileft = (
                wja * ddf_i[m, k, i, j]
                + wjb * ddf_i[m, k, i, j + 1]
                + wjc * ddddf_ij[m, k, i, j]
                + wjd * ddddf_ij[m, k, i, j + 1]
            )
            ddf_iright = (
                wja * ddf_i[m, k, i + 1, j]
                + wjb * ddf_i[m, k, i + 1, j + 1]
                + wjc * ddddf_ij[m, k, i + 1, j]
                + wjd * ddddf_ij[m, k, i + 1, j + 1]
            )

            # Then interpolate along i at theta to get the Fourier coefficient (m, k)
            f[m, k] = wia * f_ileft + wib * f_iright + wic * ddf_ileft + wid * ddf_iright

    return f


# ----------------------------------------------------------------------------
#     Calculate the Stokes vector for a given geometry (given a table of
#     Fourier coefficients read from a file).
#
#     It is assumed that the Stokes vector of the incoming sunlight
#     is [1,0,0,0], with the flux measured perpendicular to the direction
#     of incidence equal to pi.
#
#     phi and beta are assumed to be in radian
#     theta0 and theta are given as cos(theta0) and cos(theta) respectively
#
#     fou, ddf_j, ddf_i and ddddf_ij have shape (nfou, nmat, nmugs, nmugs);
#     the last three are the precomputed spline second-derivative tables for
#     fou on the xmu grid. reflection() does not check this.
#
#     Returns stokes with shape (npix, nmat + 1): row n holds the nmat Stokes
#     elements of pixel n followed by the degree of polarisation.
#
#     Pixels are independent, so the loop over them runs in parallel (prange).
#     If all pixels share the same (theta, theta0), to within SHARED_ANGLE_TOL,
#     the interpolated Fourier coefficients are computed only once (at the
#     angles of the first pixel) and reused for every pixel.
# ----------------------------------------------------------------------------
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
    nfou, nmat = fou.shape[:2]
    npix = len(phi)

    # Note, this function assumes (and does not check):
    #   - xmu is strictly increasing, with at least 2 points
    #   - fou, ddf_j, ddf_i and ddddf_ij all have the same shape (nfou, nmat, nmugs, nmugs)
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
        if not (
            abs(theta[n] - theta[0]) <= SHARED_ANGLE_TOL and abs(theta0[n] - theta0[0]) <= SHARED_ANGLE_TOL
        ):
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
        # Note: does this implicitly assume nmat=3 ?
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

        # Multiply with theta[n] (theta(n)) and the actual pixel area to obtain the stokes
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
        Read a file containing fourier coefficients produced by PyMieDAP

        Parameters:

           filename: string:

        Returns:

            nmugs: int:
               Number of gaussian integration coefficients.

            nmat: int:
               Number of matrix.

            nfou: int:
               Number of coefficients.

            rfout: array (nmugs*nmat,nmugs,nfou):
               Matrix for the fourier coefficients for reflection.

            rtra: array (nmugs*nmat,nmugs,nfou):
               Matrix for the fourier coefficients for transmission
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
                f"Expected a dense file ordered by (m, i, j), "
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
        #     m: fourier coefficient index
        #     k: matrix element index
        #   i,j: angle indices
        def fill(array):
            out = np.zeros((nfou, nmat, nmugs, nmugs))  # axes: m, k, i, j
            out[m, :, i - 1, j - 1] = array
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
        """ """
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
