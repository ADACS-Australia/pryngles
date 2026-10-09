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

    def calculate_stokes(self, phi, beta, theta0, theta, apix, qreflection=1):

        if qreflection == 1:
            table = self.rfou
        else:
            table = self.rtra

        return reflection(self.xmu, table, phi, beta, theta0, theta, apix, nmat=self.nmat, tol=1e-6)


def natural_spline_matrix(x):
    """Matrix S with y2 = S @ y for the natural cubic spline on grid x."""
    n = len(x)
    S = np.zeros((n, n))
    if n < 3:
        return S
    h = np.diff(x)
    r = np.arange(n - 2)
    # Interior tridiagonal system: A y2[1:-1] = D y
    A = np.diag((h[:-1] + h[1:]) / 3) + np.diag(h[1:-1] / 6, 1) + np.diag(h[1:-1] / 6, -1)
    D = np.zeros((n - 2, n))
    D[r, r] = 1 / h[:-1]
    D[r, r + 1] = -(1 / h[:-1] + 1 / h[1:])
    D[r, r + 2] = 1 / h[1:]
    S[1:-1] = np.linalg.solve(A, D)
    return S


def _bracket(xmu, x):
    """Bracketing indices and splint coefficients (bisect + spline_coefficients).

    Returns klo, khi and the coefficients multiplying y[klo], y[khi], y2[klo], y2[khi].
    """
    klo = np.clip(np.searchsorted(xmu, x, side="right") - 1, 0, len(xmu) - 2)
    khi = klo + 1
    h = xmu[khi] - xmu[klo]
    a = (xmu[khi] - x) / h
    b = (x - xmu[klo]) / h
    return klo, khi, a, b, (a**3 - a) * h**2 / 6, (b**3 - b) * h**2 / 6


def _reuse_map(mu, mu0, tol):
    """For each pixel, the index of the pixel whose result it reuses.

    Mirrors the C cache: a pixel reuses the last *computed* pixel if both
    cos-angles are within tol of it. tol=None disables reuse.
    """
    n = len(mu)
    rep = np.arange(n)
    if tol is None:
        return rep
    m, m0 = mu.tolist(), mu0.tolist()
    last = 0
    for i in range(1, n):
        if abs(m[i] - m[last]) < tol and abs(m0[i] - m0[last]) < tol:
            rep[i] = last
        else:
            last = i
    return rep


"""NumPy port of the C `reflection()` routine (Stam Fourier-coefficient interpolation).

Key idea: for a fixed abscissa grid `xmu`, a natural cubic spline is linear in y,
so the second derivatives are y2 = S @ y for a fixed matrix S. That lets us
  * get the second derivatives of every (k, j) row with one matmul per Fourier term,
  * turn the per-pixel second spline (along j) into a fixed weight vector W(mu),
    computed once for all Fourier terms.
"""


def reflection(xmu, table, phi, beta, theta0, theta, apix, nmat=4, tol=1e-6):
    """Stokes vector and degree of polarisation per pixel.

    xmu    : (nmugs,) ascending grid of cos(angle)
    table  : (nmugs*nmat, nmugs, nfou) -- F.rfou or F.rtra, indexed [j*nmat+k, n, m]
    phi, beta, theta0, theta, apix : (npix,) as in the C code (theta/theta0 are cosines)
    Returns (npix, nmat+1): Stokes elements times mu*apix, then P.
    """
    xmu = np.asarray(xmu, float)
    phi, beta, theta0, theta, apix = (np.asarray(v, float) for v in (phi, beta, theta0, theta, apix))
    nmugs, nfou, npix = len(xmu), table.shape[2], len(theta)

    # table[j*nmat+k, n, m] -> Rt[m, n, k, j]: n leading so gathering rows is contiguous
    Rt = np.ascontiguousarray(table.reshape(nmugs, nmat, nmugs, nfou).transpose(3, 2, 1, 0))

    # Only compute distinct geometries; the reuse pattern is the same for every m
    rep = _reuse_map(theta, theta0, tol)
    uniq, inv = np.unique(rep, return_inverse=True)

    S = natural_spline_matrix(xmu)
    lo0, hi0, a0, b0, ca0, cb0 = _bracket(xmu, theta0[uniq])
    lo, hi, a, b, ca, cb = _bracket(xmu, theta[uniq])

    # Second spline (along j, evaluated at mu) as fixed weights: value = W @ y
    W = ca[:, None] * S[lo] + cb[:, None] * S[hi]
    u = np.arange(len(uniq))
    W[u, lo] += a
    W[u, hi] += b

    RM = np.zeros((npix, nmat))
    z = np.exp(1j * phi)
    zm = np.ones(npix, complex)  # exp(i*m*phi), built by recurrence
    for m in range(nfou):
        Rm = Rt[m]  # (n, k, j)
        Q = (S @ Rm.reshape(nmugs, -1)).reshape(Rm.shape)  # y2 along n
        rf3 = np.zeros((len(uniq), nmat))
        # splint at mu0 along n, then contract with W over j
        for idx, coef, src in ((lo0, a0, Rm), (hi0, b0, Rm), (lo0, ca0, Q), (hi0, cb0, Q)):
            rf3 += coef[:, None] * np.einsum("uj,ukj->uk", W, src[idx])

        c, s = zm.real, zm.imag
        B = np.stack([c, c, s, s], axis=1)[:, :nmat]
        RM += (1.0 if m == 0 else 2.0) * B * rf3[inv]  # 2*fac, fac=0.5 for m=0
        zm *= z

    # Rotate Q, U to the reference plane
    Sv = theta0[:, None] * RM
    cb2, sb2 = np.cos(2 * beta), np.sin(2 * beta)
    q, uu = Sv[:, 1].copy(), Sv[:, 2].copy()
    Sv[:, 1] = cb2 * q + sb2 * uu
    Sv[:, 2] = -sb2 * q + cb2 * uu

    # Degree of polarisation
    I, Q_, U_ = Sv[:, 0], Sv[:, 1], Sv[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(np.abs(I) < 1e-6, 0.0, np.where(np.abs(U_) < 1e-6, -Q_ / I, np.hypot(Q_, U_) / I))
    P[np.abs(P) < 1e-6] = 0.0

    out = np.empty((npix, nmat + 1))
    out[:, :nmat] = Sv * (theta * apix)[:, None]
    out[:, nmat] = P
    return out
