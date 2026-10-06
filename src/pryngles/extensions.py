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

from pryngles.common import VERB_SIMPLE, verbose

# --------------------------------------------------
# Constants of module extensions
# --------------------------------------------------

DOUBLE = ctypes.c_double
PDOUBLE = ctypes.POINTER(DOUBLE)
PPDOUBLE = ctypes.POINTER(PDOUBLE)
PPPDOUBLE = ctypes.POINTER(PPDOUBLE)

# Load library
_spec = find_spec("pryngles.cpixx")
if _spec is None or _spec.origin is None:
    raise ImportError("pryngles.cpixx extension not built; reinstall pryngles")
cpixx_ext = ctypes.CDLL(_spec.origin)


# --------------------------------------------------
# Class ExtensionUtil
# --------------------------------------------------
class ExtensionUtil:
    """Util routines for extensions."""

    def vec2ptr(arr):
        """Converts a 1D numpy to ctypes 1D array.

        Parameters:
            arr: [ndarray] 1D numpy float64 array

        Return:
            arr_ptr: [ctypes double pointer]
        """
        arr_ptr = arr.ctypes.data_as(PDOUBLE)
        return arr_ptr

    def mat2ptr(arr):
        """Converts a 2D numpy to ctypes 2D array.

        Arguments:
            arr: [ndarray] 2D numpy float64 array

        Return:
            arr_ptr: [ctypes double pointer]

        """

        ARR_DIMX = DOUBLE * arr.shape[1]
        ARR_DIMY = PDOUBLE * arr.shape[0]

        arr_ptr = ARR_DIMY()

        # Fill the 2D ctypes array with values
        for i, row in enumerate(arr):
            arr_ptr[i] = ARR_DIMX()

            for j, val in enumerate(row):
                arr_ptr[i][j] = val

        return arr_ptr

    def ptr2mat(ptr, n, m):
        """Converts ctypes 2D array into a 2D numpy array.

        Arguments:
            arr_ptr: [ctypes double pointer]

        Return:
            arr: [ndarray] 2D numpy float64 array

        """

        arr = np.zeros(shape=(n, m))

        for i in range(n):
            for j in range(m):
                arr[i, j] = ptr[i][j]

        return arr

    def cub2ptr(arr):
        """Converts a 3D numpy to ctypes 3D array.

        Arguments:
            arr: [ndarray] 3D numpy float64 array

        Return:
            arr_ptr: [ctypes double pointer]

        """

        ARR_DIMX = DOUBLE * arr.shape[2]
        ARR_DIMY = PDOUBLE * arr.shape[1]
        ARR_DIMZ = PPDOUBLE * arr.shape[0]

        arr_ptr = ARR_DIMZ()

        # Fill the 2D ctypes array with values
        for i, row in enumerate(arr):
            arr_ptr[i] = ARR_DIMY()

            for j, col in enumerate(row):
                arr_ptr[i][j] = ARR_DIMX()

                for k, val in enumerate(col):
                    arr_ptr[i][j][k] = val

        return arr_ptr

    def ptr2cub(ptr, n, m, o):
        """Converts ctypes 3D array into a 3D numpy array.

        Arguments:
            arr_ptr: [ctypes double pointer]

        Return:
            arr: [ndarray] 3D numpy float64 array

        """

        arr = np.zeros(shape=(n, m, o))

        for i in range(n):
            for j in range(m):
                for k in range(o):
                    arr[i, j, k] = ptr[i][j][k]

        return arr


# --------------------------------------------------
# Class FourierCoefficients
# --------------------------------------------------
class FourierCoefficients(ctypes.Structure):
    """Fourier coefficients ctypes structure"""

    _fields_ = [
        ("nmat", ctypes.c_int),
        ("nmugs", ctypes.c_int),
        ("nfou", ctypes.c_int),
        ("xmu", PDOUBLE),
        ("rfou", PPPDOUBLE),
        ("rtra", PPPDOUBLE),
    ]

    def __init__(self, nmat, nmugs, nfou, xmu, rfou, rtra):
        self.nmat = nmat
        self.nmugs = nmugs
        self.nfou = nfou
        self.xmu = ExtensionUtil.vec2ptr(xmu)
        self.rfou = ExtensionUtil.cub2ptr(rfou)
        self.rtra = ExtensionUtil.cub2ptr(rtra)


# Define the argument and return types for the reflection function
cpixx_ext.reflection.restype = ctypes.c_int
cpixx_ext.reflection.argtypes = [
    FourierCoefficients,
    ctypes.c_int,
    ctypes.c_int,
    PDOUBLE,
    PDOUBLE,
    PDOUBLE,
    PDOUBLE,
    PDOUBLE,
    PPDOUBLE,
]


# --------------------------------------------------
# Class StokesScatterer
# --------------------------------------------------
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
        f = open(self.filename)

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
        f.close()

        self.nmat, self.nmugs, self.nfou = nmat, nmugs, nfou
        self.xmu, self.rfou, self.rtra = xmu, rfou, rtra
        self.F = FourierCoefficients(nmat, nmugs, nfou, xmu, rfou, rtra)

    # --------------------------------------------------
    # Tested methods from module file extensions
    # --------------------------------------------------

    def calculate_stokes(self, phi, beta, theta0, theta, apix, qreflection=1):
        """ """
        npix = len(phi)
        Sarr = np.zeros((npix, self.F.nmat + 1))
        Sarr_ptr = ExtensionUtil.mat2ptr(Sarr)
        ret = cpixx_ext.reflection(
            self.F,
            qreflection,
            npix,
            ExtensionUtil.vec2ptr(phi),
            ExtensionUtil.vec2ptr(beta),
            ExtensionUtil.vec2ptr(theta0),
            ExtensionUtil.vec2ptr(theta),
            ExtensionUtil.vec2ptr(apix),
            Sarr_ptr,
        )
        if ret != 0:
            raise RuntimeError(f"Error in reflection calculation: {ret}")
        stokes = ExtensionUtil.ptr2mat(Sarr_ptr, *Sarr.shape)
        return stokes
