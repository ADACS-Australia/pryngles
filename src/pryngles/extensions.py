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

from pryngles.common import VERB_SIMPLE, verbose

f64_1d = ndpointer(np.float64, ndim=1, flags="C_CONTIGUOUS")
f64_4d = ndpointer(np.float64, ndim=4, flags="C_CONTIGUOUS")  # rfou, rtra


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


def get_bracket_and_weights(x, xi):
    """Bracketing indices and spline weights for each xi."""
    lo = np.clip(np.searchsorted(x, xi, side="right") - 1, 0, len(x) - 2)
    hi = lo + 1
    h = x[hi] - x[lo]
    if np.any(np.abs(h) < 1e-10):
        raise RuntimeError("bad xmu grid")
    weights = ((x[hi] - xi) / h, (xi - x[lo]) / h, h)
    return lo, hi, weights


def spline_interp(yleft, yright, ddy_left, ddy_right, a, b, h):
    return a * yleft + b * yright + ((a**3 - a) * ddy_left + (b**3 - b) * ddy_right) * (h * h) / 6


def reflection(xmu, fou, ddf_j, ddf_i, ddddf_ij, phi, beta, theta0, theta, apix):
    """fou and the three tables have shape (nfou, nmat, nmugs, nmugs); returns (npix, nmat+1)."""
    nfou, nmat = fou.shape[:2]
    npix = len(phi)

    # Aliases to match naming in old C code
    mu, mu0 = theta, theta0

    ileft, iright, iweights = get_bracket_and_weights(xmu, mu)  # along i at mu
    jleft, jright, jweights = get_bracket_and_weights(xmu, mu0)  # along j at mu0

    # Interpolate a table along j at mu0, on row i (one row per pixel).
    # fou[:, :, i, j] with i, j of shape (npix,) gives shape (nfou, nmat, npix).
    def spline_interp_on_rowi(T, ddT, i):
        return spline_interp(
            T[:, :, i, jleft], T[:, :, i, jright], ddT[:, :, i, jleft], ddT[:, :, i, jright], *jweights
        )

    # f and d2f/di2 at mu0, for the two rows bracketing mu
    f_ileft = spline_interp_on_rowi(fou, ddf_j, ileft)
    f_iright = spline_interp_on_rowi(fou, ddf_j, iright)
    ddf_ileft = spline_interp_on_rowi(ddf_i, ddddf_ij, ileft)
    ddf_iright = spline_interp_on_rowi(ddf_i, ddddf_ij, iright)

    # Then along i at mu
    f = spline_interp(f_ileft, f_iright, ddf_ileft, ddf_iright, *iweights)  # (nfou, nmat, npix)

    # Sum over Fourier terms: RM = 2 * sum_m fac_m * B_k(m*phi) * f   (fac = 0.5 for m = 0)
    m = np.arange(nfou)[:, None]
    cos_m, sin_m = np.cos(m * phi), np.sin(m * phi)  # (nfou, npix)
    cos_m[0] *= 0.5  # the m = 0 fourier term only contributes half
    sin_m[0] *= 0.5
    Bplus = [cos_m, cos_m, sin_m, sin_m]

    RM = np.empty((npix, nmat))
    for k in range(nmat):
        RM[:, k] = 2 * (Bplus[k] * f[:, k]).sum(axis=0)

    # Locally reflected Stokes vector, rotated to the reference plane (assumes nmat == 3)
    SvR = mu0[:, None] * RM
    cb, sb = np.cos(2 * beta), np.sin(2 * beta)
    SvR[:, 1], SvR[:, 2] = (cb * SvR[:, 1] + sb * SvR[:, 2], -sb * SvR[:, 1] + cb * SvR[:, 2])

    # Degree of polarisation
    S0, S1, S2 = SvR[:, 0], SvR[:, 1], SvR[:, 2]
    zero0 = np.abs(S0) < 1e-6
    S0 = np.where(zero0, 1.0, S0)  # avoid dividing by ~0
    P = np.where(np.abs(S2) < 1e-6, -S1 / S0, np.hypot(S1, S2) / S0)
    P = np.where(zero0 | (np.abs(P) < 1e-6), 0.0, P)

    stokes = np.empty((npix, nmat + 1))
    stokes[:, :nmat] = SvR * (mu * apix)[:, None]
    stokes[:, nmat] = P
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
