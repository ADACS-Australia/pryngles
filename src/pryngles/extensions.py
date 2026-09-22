##################################################################
#                                                                #
#.#####...#####...##..##..##..##...####...##......######...####..#
#.##..##..##..##...####...###.##..##......##......##......##.....#
#.#####...#####.....##....##.###..##.###..##......####.....####..#
#.##......##..##....##....##..##..##..##..##......##..........##.#
#.##......##..##....##....##..##...####...######..######...####..#
#................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# External required packages
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from pryngles import *

import ctypes
import glob

#Load library
libfile = glob.glob(Misc.get_data('../cpixx*.so'))[0]
cpixx_ext=ctypes.CDLL(libfile)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Stand alone code of the module
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Calculate reflection
cpixx_ext.reflection.restype = ctypes.c_int
cpixx_ext.reflection.argtypes = [
    ctypes.Structure,
    ctypes.c_int,
    ctypes.c_int,
    PDOUBLE,PDOUBLE,PDOUBLE,PDOUBLE,PDOUBLE,
    PPDOUBLE
]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class ExtensionUtil
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ExtensionUtil(object):
    """Util routines for extensions.
    """
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
        """ Converts a 2D numpy to ctypes 2D array. 

        Arguments:
            arr: [ndarray] 2D numpy float64 array

        Return:
            arr_ptr: [ctypes double pointer]

        """

        ARR_DIMX = DOUBLE*arr.shape[1]
        ARR_DIMY = PDOUBLE*arr.shape[0]

        arr_ptr = ARR_DIMY()

        # Fill the 2D ctypes array with values
        for i, row in enumerate(arr):
            arr_ptr[i] = ARR_DIMX()

            for j, val in enumerate(row):
                arr_ptr[i][j] = val

        return arr_ptr

    def ptr2mat(ptr, n, m):
        """ Converts ctypes 2D array into a 2D numpy array. 

        Arguments:
            arr_ptr: [ctypes double pointer]

        Return:
            arr: [ndarray] 2D numpy float64 array

        """

        arr = np.zeros(shape=(n, m))

        for i in range(n):
            for j in range(m):
                arr[i,j] = ptr[i][j]

        return arr

    def cub2ptr(arr):
        """ Converts a 3D numpy to ctypes 3D array. 

        Arguments:
            arr: [ndarray] 3D numpy float64 array

        Return:
            arr_ptr: [ctypes double pointer]

        """

        ARR_DIMX = DOUBLE*arr.shape[2]
        ARR_DIMY = PDOUBLE*arr.shape[1]
        ARR_DIMZ = PPDOUBLE*arr.shape[0]

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
        """ Converts ctypes 3D array into a 3D numpy array. 

        Arguments:
            arr_ptr: [ctypes double pointer]

        Return:
            arr: [ndarray] 3D numpy float64 array

        """

        arr = np.zeros(shape=(n, m, o))

        for i in range(n):
            for j in range(m):
                for k in range(o):
                    arr[i,j,k] = ptr[i][j][k]

        return arr



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class FourierCoefficients
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class FourierCoefficients(ctypes.Structure):
    """Fourier coefficients ctypes structure
    """
    _fields_=[
        ("nmat",ctypes.c_int),
        ("nmugs",ctypes.c_int),
        ("nfou",ctypes.c_int),
        ("xmu",PDOUBLE),
        ("rfou",PPPDOUBLE),
        ("rtra",PPPDOUBLE),
    ]
    def __init__(self,nmat,nmugs,nfou,xmu,rfou,rtra):
        self.nmat=nmat
        self.nmugs=nmugs
        self.nfou=nfou
        self.xmu=ExtensionUtil.vec2ptr(xmu)
        self.rfou=ExtensionUtil.cub2ptr(rfou)
        self.rtra=ExtensionUtil.cub2ptr(rtra)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class StokesScatterer
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class StokesScatterer(object):
    """Stokes scatterer
    """
    
    def __init__(self,filename):
        self.filename=filename
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
        f=open(self.filename)

        #Read header
        nmat=0
        imu=0
        for i,line in enumerate(f):
            if '#' in line:
                continue
            data=line.split()
            if len(data)<3:
                if len(data)==1:
                    if not nmat:
                        nmat=int(data[0])
                    else:
                        nmugs=int(data[0])
                        xmu=np.zeros(nmugs)
                else:
                    xmu[imu]=float(data[0])
                    imu+=1
            else:
                break

        #Get core data
        data=np.loadtxt(self.filename,skiprows=i)
        nfou=int(data[:,0].max())+1

        rfou=np.zeros((nmat*nmugs,nmugs,nfou))
        rtra=np.zeros((nmat*nmugs,nmugs,nfou))

        #Read fourier coefficients
        for row in data:
            m,i,j=int(row[0]),int(row[1])-1,int(row[2])-1
            ibase=i*nmat
            rfou[ibase:ibase+3,j,m]=row[3:3+nmat]
            if len(row[3:])>nmat:
                rtra[ibase:ibase+3,j,m]=row[3+nmat:3+2*nmat]

        verbose(VERB_SIMPLE,f"Checksum '{self.filename}': {rfou.sum()+rtra.sum():.16e}")
        f.close()
        
        self.nmat,self.nmugs,self.nfou=nmat,nmugs,nfou
        self.xmu,self.rfou,self.rtra=xmu,rfou,rtra
        self.F=FourierCoefficients(nmat,nmugs,nfou,xmu,rfou,rtra)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Tested methods from module file extensions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def calculate_stokes(self,phi,beta,theta0,theta,apix,qreflection=1):
        """
        """
        npix=len(phi)
        Sarr=np.zeros((npix,self.F.nmat+1))

        use_python = True

        if not use_python:
            Sarr_ptr=ExtensionUtil.mat2ptr(Sarr)
            cpixx_ext.reflection(self.F,qreflection,npix,
                                ExtensionUtil.vec2ptr(phi),
                                ExtensionUtil.vec2ptr(beta),
                                ExtensionUtil.vec2ptr(theta0),
                                ExtensionUtil.vec2ptr(theta),
                                ExtensionUtil.vec2ptr(apix),
                                Sarr_ptr)
            stokes=ExtensionUtil.ptr2mat(Sarr_ptr,*Sarr.shape)
        else:
            stokes = reflection(self.nmat, self.nmugs, self.nfou, # integers
                                self.rfou, self.rtra,             # 3D arrays
                                self.xmu,                         # 1D array
                                qreflection, npix,                # integers
                                phi, beta, theta0, theta, apix    # 1D arrays
                                )
        return stokes

import sys
import numpy as np
from numba import njit, int64, float64, types

@njit(float64[:](float64[:], float64[:], int64))
def spline(x, y, n):
    u = np.zeros(n)
    y2 = np.zeros(n)

    y2[0] = 0
    u[0] =  0

    for i in range(1, n-1):
        sig = (x[i]-x[i-1])/(x[i+1]-x[i-1])
        p = sig*y2[i-1]+2
        y2[i] = (sig-1)/p
        u[i] = (6*((y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/
                (x[i]-x[i-1]))/(x[i+1]-x[i-1]) - sig*u[i-1])/p

    qn = 0
    un = 0
    y2[n-1] = (un-qn*u[n-2])/(qn*y2[n-2]+1)

    for k in range(n-2, -1, -1):
        y2[k] = y2[k]*y2[k+1]+u[k]

    return y2

@njit(int64(float64[:], int64, float64))
def bisect(xa, n ,x):
    klo = 0
    khi = n - 1

    while ((khi - klo) > 1):
        k = int((klo + khi) / 2) # Integer division
        if (xa[k] > x):
            khi = k
        else:
            klo = k

    return klo

@njit(types.Tuple((int64, int64, float64, float64, float64))(float64[:], int64, float64))
def spline_coefficients(xa, n, x):
    # Find indices to interpolate between
    klo = bisect(xa, n, x)
    khi = klo + 1

    h = xa[khi] - xa[klo]

    if (np.abs(h) < 1e-10):
        print("ERROR in spline_coefficients: bad xa input.")
        # Should we exit with error here?

    # Coefficients
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h

    return klo, khi, a, b, h


@njit(float64( float64[:], float64[:], int64, int64, float64, float64, float64) )
def splint(ya, y2a, klo, khi, a, b, h):
    y = a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6
    return y

@njit(float64[:,:](
        int64, int64, int64,
        float64[:,:,:], float64[:,:,:], float64[:],
        int64, int64,
        types.Array(dtype=types.float64, ndim=1, layout='C', readonly=True), # phi is readonly, so can't use just float64[:] here
        float64[:], float64[:], float64[:], float64[:],
        ))
def reflection(nmat, nmugs, nfou,
                rfou, rtra, xmu,
                qreflection, npix,
                phi,
                beta, theta0, theta, apix): 
    
    # This function returns the following 2D array
    Sarr = np.zeros((npix, nmat+1))

    # Cubes
    rf = np.zeros((nmat, nmugs, nmugs))
    rfsec = np.zeros((nmat, nmugs, nmugs))

    # Matrices
    rfmu0 = np.zeros((nmat, nmugs))
    RM = np.zeros((npix, nmat))

    # Vectos nmugs
    rfsecmu0 = np.zeros(nmugs)

    # Vectos nmats
    rf3save = np.zeros(nmat)
    SvR = np.zeros(nmat)

    Bplus = np.zeros(4)
    muold = 1
    mu0old = 1

    # Loop over the Fourier coefficients:
    for m in range(nfou):

        fac = 1.0
        if (m == 0):
            fac = 0.5

        # Initialize the interpolation matrix for the current fourier coefficient:
        for j in range(nmugs):
            for k in range(nmat):
                ki = j * nmat + k
                for n in range(nmugs):
                    if (qreflection == 1):
                        rf[k,j,n] = rfou[ki,n,m]
                    else:
                        rf[k,j,n] = rtra[ki,n,m]


                # Use slice rf(k,j,:), write directly into corresponding rfsec row
                rfsec[k,j,:] = spline(xmu, rf[k,j,:], nmugs)

        #/*
        # *----------------------------------------------------------------------------
        # *     Loop over the pixels:
        # *       If the input angles are (very) similar to a previously calculated
        # *       case use those values. To obtain obtain the fourier coefficient at
        # *       (mu,mu0) spline has to be called a second time.
        # *----------------------------------------------------------------------------
        # */
        for i in range(npix):
            mu = theta[i]
            mu0 = theta0[i]
            Bplus[0] = np.cos(m * phi[i])
            Bplus[1] = np.cos(m * phi[i])
            Bplus[2] = np.sin(m * phi[i])
            Bplus[3] = np.sin(m * phi[i])

            if ((i > 0) and ((np.abs(mu - muold) < 1e-6)) and (np.abs(mu0 - mu0old) < 1e-6)):

                for k in range(nmat):
                    RM[i,k] = RM[i,k] + 2 * Bplus[k] * fac * rf3save[k]

            else:

                # Get indices and coefficients to work with (at mu0)
                klo, khi, a, b, h = spline_coefficients(xmu, nmugs, mu0)

                for j in range(nmugs):
                    for k in range(nmat):
                        # Use rf(k,j,:) and rfsec(k,j,:) slices directly, and write
                        # straight into rmu0(k,j,:) Do the spline interpolation between klo
                        # and khi, given the coefficients a,b,h
                        rfmu0[k,j] = splint(rf[k,j,:], rfsec[k,j,:], klo, khi, a, b, h)

                # Get indices and coefficients to work with (now at mu)
                klo, khi, a, b, h = spline_coefficients(xmu, nmugs, mu)

                for k in range(nmat):
                    rfsecmu0 = spline(xmu, rfmu0[k,:], nmugs)
                    # Do the spline interpolation between klo and khi, given the coefficients a,b,h
                    rf3save[k] = splint(rfmu0[k,:], rfsecmu0, klo, khi, a, b, h)
                    muold = mu
                    mu0old = mu0
                    RM[i,k] = RM[i,k] + 2 * Bplus[k] * fac * rf3save[k]

    # Loop again over the pixels to rotate Stokes vector:
    for i in range(npix):
        mu = theta[i]
        mu0 = theta0[i]

        # Calculate the locally reflected Stokes vector:
        for k in range(nmat):
            SvR[k] = mu0 * RM[i,k]

        # Rotate Stokes elements Q and U to the actual reference plane:
        be = 2 * beta[i]
        SvR2 = np.cos(be) * SvR[1] + np.sin(be) * SvR[2]
        SvR3 = -np.sin(be) * SvR[1] + np.cos(be) * SvR[2]
        SvR[1] = SvR2
        SvR[2] = SvR3

        # Compute the local degree of polarisation P:
        if (np.abs(SvR[0]) < 1e-6):
            P = 0
        elif (np.abs(SvR[2]) < 1e-6):
            P = -SvR[1] / SvR[0]
        else:
            P = np.sqrt(SvR[1] * SvR[1] + SvR[2] * SvR[2]) / SvR[0]

        if (np.abs(P) < 1e-6):
            P = 0

        #/*
        # *----------------------------------------------------------------------------
        # * Add the Stokes elements of the pixel to an array:
        # *   Multiply with mu and the actual pixel area to obtain stokes elements
        # *----------------------------------------------------------------------------
        # */
        for k in range(nmat):
            Sarr[i,k] = SvR[k] * mu * apix[i]

        # The value of the degree of polarization
        Sarr[i,nmat] = P

    return Sarr # 2D array
