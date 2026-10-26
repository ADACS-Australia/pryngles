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
f64_2d = ndpointer(np.float64, ndim=2, flags="C_CONTIGUOUS")  # Sarr
f64_4d = ndpointer(np.float64, ndim=4, flags="C_CONTIGUOUS")  # rfou, rtra


# Load library
_spec = find_spec("pryngles.cpixx")
if _spec is None or _spec.origin is None:
    raise ImportError("pryngles.cpixx extension not built; reinstall pryngles")
cpixx_ext = ctypes.CDLL(_spec.origin)


# Define the argument and return types for the reflection function
cpixx_ext.reflection.restype = ctypes.c_int
cpixx_ext.reflection.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    f64_1d,
    f64_4d,
    f64_4d,
    f64_4d,
    f64_4d,
    ctypes.c_int,
    f64_1d,
    f64_1d,
    f64_1d,
    f64_1d,
    f64_1d,
    f64_2d,
]

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

        ret = cpixx_ext.reflection(
            self.nmat,
            self.nmugs,
            self.nfou,
            self.xmu,
            fourier_coefficients,
            fourier_derivs["j"],
            fourier_derivs["i"],
            fourier_derivs["ij"],
            npix,
            phi,
            beta,
            theta0,
            theta,
            apix,
            stokes,
        )

        if ret != 0:
            raise RuntimeError(f"Error in reflection calculation: {ret}")

        return stokes
