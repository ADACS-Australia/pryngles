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
f64_3d = ndpointer(np.float64, ndim=3, flags="C_CONTIGUOUS")  # rfou, rtra


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
    f64_3d,
    ctypes.c_int,
    f64_1d,
    f64_1d,
    f64_1d,
    f64_1d,
    f64_1d,
    f64_2d,
]


class StokesScatterer:
    """Stokes scatterer"""

    def __init__(self, filename):
        self.filename = filename
        self.read_fourier()

    def read_fourier(self):
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
        with open(self.filename, "r") as f:

            # Read header
            nmat = 0
            imu = 0
            for _i, line in enumerate(f):
                if "#" in line:
                    continue
                data = line.split()
                if len(data) < 3:
                    if len(data) == 1:
                        if not nmat:
                            nmat = int(data[0])
                        else:
                            nmugs = int(data[0])
                            xmu = np.zeros(nmugs)
                    else:
                        xmu[imu] = float(data[0])
                        imu += 1
                else:
                    break

        # Get core data
        data = np.loadtxt(self.filename, skiprows=_i)
        nfou = int(data[:, 0].max()) + 1

        rfou = np.zeros((nmat * nmugs, nmugs, nfou))
        rtra = np.zeros((nmat * nmugs, nmugs, nfou))

        # Read fourier coefficients
        for row in data:
            m, i, j = int(row[0]), int(row[1]) - 1, int(row[2]) - 1
            ibase = i * nmat
            rfou[ibase : ibase + 3, j, m] = row[3 : 3 + nmat]
            if len(row[3:]) > nmat:
                rtra[ibase : ibase + 3, j, m] = row[3 + nmat : 3 + 2 * nmat]

        verbose(VERB_SIMPLE, f"Checksum '{self.filename}': {rfou.sum() + rtra.sum():.16e}")

        self.nmat, self.nmugs, self.nfou = nmat, nmugs, nfou
        self.xmu, self.rfou, self.rtra = xmu, rfou, rtra

    def calculate_stokes(self, phi, beta, theta0, theta, apix, qreflection=1):
        """ """
        npix = len(phi)
        stokes = np.zeros((npix, self.nmat + 1))
        fourier_coefficients = self.rfou if qreflection else self.rtra
        ret = cpixx_ext.reflection(
            self.nmat,
            self.nmugs,
            self.nfou,
            self.xmu,
            fourier_coefficients,
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
