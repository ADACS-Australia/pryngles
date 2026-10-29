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

import ctypes
from importlib.util import find_spec

import numpy as np
from numpy.ctypeslib import ndpointer
from numba import njit, types

from pryngles.common import VERB_SIMPLE, verbose

f64_1d = ndpointer(np.float64, ndim=1, flags="C_CONTIGUOUS")
f64_4d = ndpointer(np.float64, ndim=4, flags="C_CONTIGUOUS")  # rfou, rtra

# Plain numba types
f64 = types.float64
i64 = types.int64

# Read-only C-contiguous float64 numba array types
f64_arr1 = types.Array(f64, 1, "C", readonly=True)
f64_arr4 = types.Array(f64, 4, "C", readonly=True)


# Load library
_spec = find_spec("pryngles.cpixx")
if _spec is None or _spec.origin is None:
    raise ImportError("pryngles.cpixx extension not built; reinstall pryngles")
cpixx_ext = ctypes.CDLL(_spec.origin)

cpixx_ext.get_spline_derivs.restype = ctypes.c_int
cpixx_ext.get_spline_derivs.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    f64_1d,
    f64_4d,
    f64_4d,
    f64_4d,
    f64_4d,
]


def calculate_spline_derivs(xmu, fou):
    nfou, nmat, nmugs, nmugs2 = fou.shape
    if nmugs != nmugs2:
        raise RuntimeError("fourier array not square")
    ddf_j = np.zeros_like(fou)
    ddf_i = np.zeros_like(fou)
    ddddf_ij = np.zeros_like(fou)
    ret = cpixx_ext.get_spline_derivs(nmat, nmugs, nfou, xmu, fou, ddf_j, ddf_i, ddddf_ij)
    if ret != 0:
        raise RuntimeError(f"Error in get_spline_derivs: {ret}")
    return {"j": ddf_j, "i": ddf_i, "ij": ddddf_ij}


@njit(i64(f64_arr1, f64), cache=True)
def bisect(x, xi):
    klo = 0
    khi = len(x) - 1

    while (khi - klo) > 1:
        k = (klo + khi) // 2
        if x[k] > xi:
            khi = k
        else:
            klo = k

    return klo


#
# Find the bracketing indices and interpolation coefficients for xi in x.
# x must be strictly increasing with at least 2 points.
#
@njit(types.Tuple((i64, i64, f64, f64, f64))(f64_arr1, f64), cache=True)
def spline_coefficients(x, xi):
    # Find indices to interpolate between
    klo = bisect(x, xi)
    khi = klo + 1

    h = x[khi] - x[klo]

    # Coefficients
    a = (x[khi] - xi) / h
    b = (xi - x[klo]) / h

    return klo, khi, a, b, h


#
# ----------------------------------------------------------------------------
#     Spline interpolation routine from Press et al. (1986, p.88).
#
#     Given the arrays y and ddy (second derivatives of y), the precomputed
#     lower and upper indices klo/khi and spline coefficients a, b, h (from
#     spline_coefficients), returns the cubic-spline interpolated value.
# ----------------------------------------------------------------------------
#
@njit(f64(f64_arr1, f64_arr1, i64, i64, f64, f64, f64), cache=True)
def interp(y, ddy, klo, khi, a, b, h):
    return a * y[klo] + b * y[khi] + (
        (a * a * a - a) * ddy[klo] + (b * b * b - b) * ddy[khi]
    ) * (h * h) / 6


# Maximum number of Stokes elements reflection() supports (size of Bplus etc.)
MAX_NSTOKES = 4
REFLECTION_TOLERANCE = 1.0e-6

#
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
# ----------------------------------------------------------------------------
#
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
)
def reflection(xmu, fou, ddf_j, ddf_i, ddddf_ij, phi, beta, theta0, theta, apix):
    nfou, nmat = fou.shape[:2]
    npix = len(phi)

    muold = 1.0
    mu0old = 1.0

    stokes = np.zeros((npix, nmat + 1))

    # Temporary arrays (numba has no stack arrays, so everything is allocated
    # once up front, including the two-element scratch arrays used by interp)
    RM = np.zeros((npix, nmat))
    Bplus = np.zeros(MAX_NSTOKES)
    f_mu0_mu = np.zeros(MAX_NSTOKES)
    SvR = np.zeros(MAX_NSTOKES)
    f_mu0 = np.zeros(2)
    ddf_mu0 = np.zeros(2)

    # Loop over the Fourier coefficients:
    for m in range(nfou):

        fac = 0.5 if m == 0 else 1.0

        # ----------------------------------------------------------------------
        #     Loop over the pixels.
        #
        #     If the input angles are (very) similar to a previously calculated
        #     case, reuse those values.
        #
        #     To get the fourier coefficient at (mu, mu0), interpolate along j at
        #     mu0 for the two rows bracketing mu, then along i at mu. The second
        #     pass needs d2f/di2 at those rows; it comes from interpolating the
        #     precomputed ddf_i along j (spline construction along i and
        #     interpolation along j commute), so no spline is solved per pixel.
        # ----------------------------------------------------------------------
        for n in range(npix):
            mu = theta[n]
            mu0 = theta0[n]
            cosmphi = np.cos(m * phi[n])
            sinmphi = np.sin(m * phi[n])
            Bplus[0] = cosmphi
            Bplus[1] = cosmphi
            Bplus[2] = sinmphi
            Bplus[3] = sinmphi

            if (
                n <= 0
                or not (abs(mu - muold) < REFLECTION_TOLERANCE)
                or not (abs(mu0 - mu0old) < REFLECTION_TOLERANCE)
            ):
                # Interpolate the fourier terms at j=mu0 and i=mu, otherwise skip
                # and reuse previous cached value

                # Get indices and coefficients to work with (at mu0, along j)
                jlo, jhi, ja, jb, jh = spline_coefficients(xmu, mu0)

                # Get indices and coefficients to work with (now at mu, along i)
                ilo, ihi, ia, ib, ih = spline_coefficients(xmu, mu)

                for k in range(nmat):
                    # Only the two rows i=ilo,ihi that bracket mu are needed. For
                    # each, interpolate f and d2f/di2 along j at mu0, given the
                    # coefficients ja,jb,jh
                    f_mu0[0] = interp(fou[m, k, ilo], ddf_j[m, k, ilo], jlo, jhi, ja, jb, jh)
                    f_mu0[1] = interp(fou[m, k, ihi], ddf_j[m, k, ihi], jlo, jhi, ja, jb, jh)
                    ddf_mu0[0] = interp(ddf_i[m, k, ilo], ddddf_ij[m, k, ilo], jlo, jhi, ja, jb, jh)
                    ddf_mu0[1] = interp(ddf_i[m, k, ihi], ddddf_ij[m, k, ihi], jlo, jhi, ja, jb, jh)
                    # Do the spline interpolation along i between the two rows,
                    # given the coefficients ia,ib,ih
                    f_mu0_mu[k] = interp(f_mu0, ddf_mu0, 0, 1, ia, ib, ih)
                muold = mu
                mu0old = mu0

            # Compute the reflection matrix at pixel n and stokes parameter k,
            # summing over fourier terms m
            for k in range(nmat):
                RM[n, k] += 2 * Bplus[k] * fac * f_mu0_mu[k]

    # Loop again over the pixels to rotate the Stokes vector
    for n in range(npix):
        mu = theta[n]
        mu0 = theta0[n]

        # Calculate the locally reflected Stokes vector:
        for k in range(nmat):
            SvR[k] = mu0 * RM[n, k]

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

        # Multiply with mu and the actual pixel area to obtain the stokes
        # element k for each pixel n.
        for k in range(nmat):
            stokes[n, k] = SvR[k] * mu * apix[n]

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

        # Don't rely on C extension safely handling zero-length arrays
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
