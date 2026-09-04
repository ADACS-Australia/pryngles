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
import numpy as np
import math as mh
import spiceypy as spy
from scipy.integrate import quad
from scipy.spatial import ConvexHull
from celluloid import Camera # getting the camera
import rebound as rb
import matplotlib.pyplot as plt


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class Science
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Science(PrynglesCommon):
    """
    The Science utility Class defines all the methods to geometric and scientific useful calculations. 
    It is a static Class, i.e., that cannot be instantiated and their methods work as regular functions.
    """

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Tested methods from module file science
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def spherical(xyz):
        """
        Transform cartesian coordinates into spherical coordinates
    
        Parameters
        ----------
        xyz : `np.array(N = 3)`
            Array with components of Cartesian Coordinates 
    
        Returns
        -------
        :
            rqf : `np.array(N = 3)`
                Spherical coordinates :math:`(r, θ, \phi)` where :math:`θ` is azimutal angle and :math:`\phi` is 
                elevation (complement of polar angle).

        Examples
        ------------
        >>> xyz = [1, 1, 1]
        >>> 
        >>> # To Spherical
        >>> pr.Science.spherical(xyz)
        array([1.73205081, 0.78539816, 0.61547971])
        """
        r,theta,phi=spy.reclat(np.array(xyz))
        theta=2*mh.pi+theta if theta<0 else theta
    
        return np.array([r,theta,phi])
    
    def cospherical(xyz):
        """Transform cartesian coordinates into cosine/sine of spherical angles
        
        Parameters
        ----------
        xyz : `np.array(N = 3)`
            Array with components of Cartesian Coordinates 
    
        Returns
        -------
        :
            cqsqcf : `np.array (N = 3)`
                Cosine/Sine of spherical angles (:math:`\cos θ`, :math:`\sin θ`, :math:`\sin\phi`) where :math:`θ` is 
                azimutal angle and :math:`\phi` is elevation (complement of polar angle).   

        Examples
        ------------
        >>> xyz = [1, 1, 1]
        >>> 
        >>> # To Cospherical
        >>> pr.Science.cospherical(xyz)
        array([0.70710678, 0.70710678, 0.57735027])     
        """
        rho=(xyz[0]**2+xyz[1]**2)**0.5
        sf=xyz[2]/(rho**2+xyz[2]**2)**0.5
        cq=xyz[0]/rho if not mh.isclose(rho,0) else 1
        sq=xyz[1]/rho if not mh.isclose(rho,0) else 0
        return np.array([cq,sq,sf])
    
    def pcylindrical(xyz):
        """Transform cartesian coordinates into pseudo cylindrical coordinates
        
        Parameters
        ----------
        xyz : `np.array`
            Array with components of Cartesian Coordinates 
                
        Returns
        ------------
        :
            rhoazcf : `np.array`
                Cylindrical coordinates expresed as :math:`\\rho, \phi` (azimutal angle) and :math:`\cos\\theta` (cosine
                of polar angle). 

        Examples
        ------------
        >>> xyz = [1, 1, 1]
        >>> 
        >>> # To pseudoCylindrical
        >>> pr.Science.pcylindricak(xyz)
        array([1.41421356, 0.78539816, 0.57735027]) 
        """
        xyz = np.asarray(xyz)

        # Check if single input or multiple inputs
        single_input = xyz.ndim == 1
        if single_input:
            xyz = xyz[np.newaxis, :]  # convertir a (1, 3)

        # Cartesian components
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        rho = np.sqrt(x**2 + y**2)
        r = np.sqrt(rho**2 + z**2)
        phi = np.arctan2(y, x)
        phi = np.where(phi >= 0, phi, 2 * np.pi + phi)
        cost = np.where(np.isclose(r, 0), np.copysign(1, z), z / r)

        result = np.column_stack((rho, phi, cost))

        return result[0] if single_input else result

    
    def cartesian(rqf):
        """
        Transform spherical coordinates into cartesian coordinates
    
        Parameters
        ----------
        rqf : `np.array(N = 3)`
            Spherical coordinates :math:`(r, θ, \phi)` where :math:`θ` is azimutal angle and :math:`\phi` is 
            elevation (complement of polar angle).
                
        Returns
        ------------
        :
            xyz : `np.array(N = 3)`
                Array with components of Cartesian Coordinates

        Examples
        ------------
        >>> xyz = [1, 1, 1]
        >>> 
        >>> # From Spherical to Cartesian
        >>> pr.Science.cartesian([1, 30*pr.Consts.deg, 60*pr.Consts.deg])
        array([0.4330127, 0.25     , 0.8660254])
        """
        return spy.latrec(rqf[0],rqf[1],rqf[2])

    def direction(*args):
        """Calculate the direction on which a vector is pointing
        
        Parameters
        --------------
            args : `np.array`
                If ``len(args) == 2``, components are longitude and latitude (:math:`\lambda,\\beta`) of the direction (in degrees).
                If ``len(args) == 3``, components are cartisian coordinates of the vector :math:`\hat{n}`.
                
        Returns
        ------------
        :
            nx, ny, nz: `float`
                Cartesian components (if ``len(args) == 2``) of the unitary direction vector :math:`\hat{n}`.
    
            lamb, beta: `float` [deg]
                If ``len(args) == 3``, Longitude and Latitude (angle with respect to x-axis & elevation angle with respect to xy-plane)

        Raises
        ----------
        ValueError
            In case you provided a wrong number of arguments ``{args}``. It should be 2 or 3 arguments
        ValueError
            If Latitude valur for elevation angle should be is not in the interval [-90, 90]

        Examples
        -----------
        >>> # Get the Unitary Vector in the direction long = 30, lat = 60
        >>> Science.direction(30,60)
        array([0.4330127, 0.25, 0.8660254])

        >>> # Get the Long, Lat values for a direction vector
        >>> Science.direction(1, 1, 0)
        (45.0, 0.0)
        """
        if len(args)==3:
            rqf=Science.spherical(list(args))
            return rqf[1]*Consts.rad,rqf[2]*Consts.rad
        elif len(args)==2:
            if abs(args[1])>90:
                raise ValueError("Elevation angle should be in the interval [-90,90]")
            nvec=Science.cartesian([1,args[0]*Consts.deg,args[1]*Consts.deg])
            return nvec
        else:
            raise ValueError("You provided a wrong number of arguments '{args}'.  It should be 2 or 3'")
    
    def rotation_matrix(ez,alpha):
        """
        Set a rotation matrix from the direction of the :math:`\hat{e_z}` vector and a rotation angle :math:`\\alpha`
        
        Parameters
        -----------------------
        ez : `np.array(N = 3)`
            Vector in the direction of the z-axis. 
                
        alpha : `float` [rad]
            Rotation angle of the x-axis around z-axis (clockwise)
                
        Returns
        ---------------
        :
            Msys2uni : `np.array(3x3)`
                Rotation matrix from the system defined by :math:`\hat{e_z}` and the universal system.
                
            Muni2sys : `np.array(3x3)`
                Rotation matrix from the universal system to the system defined by :math:`\hat{e_z}`
        
        Note
        ----------------
        Universal System refers to the euclidean base :math:`\{\hat{e_i}\}`

        Examples
        -----------
        >>> # Rotation matrices from the direction of the z-axis and a roll in the x-y plane
        >>> Msys2uni, Muni2sys = Science.rotation_matrix([0, 0, 1], 0)
        >>> print(Msys2uni)
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]
        """
        ez,one=spy.unorm(ez)
        ex=spy.ucrss([0,0,1],ez) #Spice is 5 faster for vcrss
        if spy.vnorm(ex)==0:
            ex=np.array([1,0,0]) if np.sum(ez)>0 else np.array([-1,0,0])
        ey=spy.ucrss(ez,ex)
        Msys2uni=np.array(list(np.vstack((ex,ey,ez)).transpose())).reshape((3,3))
        Muni2sys=spy.invert(Msys2uni)
        verbose(VERB_VERIFY,"Rotation axis:",ex,ey,ez)
        return Msys2uni,Muni2sys
    
    def limb_darkening(rho, cs = [0.6562], N = None):
        """
        Non-Linear model for Limb-Darkening computation
        
        Parameters
        -----------------
        rho : `float`
            Distance to center of the star in units of stellar radius.
            
        cs : `list`
            List of limb darkening coefficients. Its lenght determines the model to use between Linear, Quadratic and 3/4 Parameter Non-Linear | **Default** = `[0.6562]`
            
        N : `float`
            Normalization constant | **Default** = `None`
            
        Returns
        -------------
        :
            I : `float`
            Normalized intensity of the star at projected distance :math:`\\rho`.

        Warnings
        -------------
        This method will replace :any:`legacy.Util.limbDarkening` for versions > 0.10.x

        Raises
        ------------------
        ValueError
            If `len(cs) > 4`. Limb darkening model not implemented for order greater than 4
        
        Note
        -------------
        This method implements the non-linear limb-darkening model described in **[1]** and implements limb darkening coefficients from **[2]**.

        .. math::

            \\mu = (1-\\rho^2)^{1/2}

        .. math::

            \\frac{I(\\mu)}{I(0)} = 1-\\sum_n c_n(1-\\mu^{n/2})

        **[1]** Models in: https://pages.jh.edu/~dsing3/David_Sing/Limb_Darkening.html

        **[2]** Coefficients available at: https://pages.jh.edu/~dsing3/LDfiles/LDCs.CoRot.Table1.txt

        Examples
        -------------
        >>> # Creating a Figure Plot
        >>> fig = plt.figure()
        >>> ax = fig.gca()
        >>>
        >>> # Projected distance to Star center
        >>> rhos = np.linspace(0, 1, 100) # [R_star] units
        >>> 
        >>> # Linear Model Computation
        >>> coefs = [0.6550] 
        >>> ax.plot(rhos, Science.limb_darkening(rhos,coefs), label = f"coefs = {coefs}")
        >>> 
        >>> # Quadratic Model Computation 
        >>> coefs = [0.6022, 0.0654]
        >>> ax.plot(rhos, Science.limb_darkening(rhos, coefs), label = f"coefs = {coefs}")
        >>> 
        >>> # 3 Parameter Non-Linear Model Computation 
        >>> coefs=[0.9724, -0.4962, 0.2029]
        >>> ax.plot(rhos,Science.limb_darkening(rhos,coefs),label=f"coefs = {coefs}")    
        >>> 
        >>> # 4 Parameter Non-Linear Model Computation 
        >>> coefs=[-0.2018, 2.1000, -2.0247, 0.7567]
        >>> ax.plot(rhos, Science.limb_darkening(rhos, coefs), label = f"coefs = {coefs}")
        >>>
        >>> # For decoration purposes
        >>> ax.legend()
        >>> ax.set_xlabel(r"$\\rho$")
        >>> ax.set_ylabel(r"$I(\\rho)/I(0)$")

        .. image:: images/science_limb.png
            :align: center
            :width: 600px     
        """
        mu=(1-rho**2)**0.5
        order=len(cs)
        
        #Calculate normalization constant
        if N is None:
            chash=hash(tuple(cs))
            if chash in SCIENCE_LIMB_NORMALIZATIONS:
                N=SCIENCE_LIMB_NORMALIZATIONS[chash]
            else:
                integrand=lambda rho:Science.limb_darkening(rho,cs,N=1)*2*np.pi*rho
                N=quad(integrand,0.0,1.0,epsrel=1e-5)[0]
                verbose(VERB_VERIFY,f"Normalization of limb darkening function for cs = {cs}, N = {N}")
                SCIENCE_LIMB_NORMALIZATIONS[chash]=N
                
        if order==0:
            I=np.ones_like(rho)
        elif order==1:
            I=1-cs[0]*(1-mu)
        elif order==2:
            I=1-cs[0]*(1-mu)-cs[1]*(1-mu)**2
        elif order==3:
            I=1-cs[0]*(1-mu)-cs[1]*(1-mu**1.5)-cs[2]*(1-mu**2)
        elif order==4:
            I=1-cs[0]*(1-mu**0.5)-cs[1]*(1-mu)-cs[2]*(1-mu**1.5)-cs[3]*(1-mu**2)
        else:
            raise ValueError(f"Limb darkening not implemented for order {order}")
        return I/N
    
    def blackbody_intensity(wavelength, temperature):
        """
        Planck's Blackbody distribution function for Intensity :math:`B_λ`
        
        :math:`B_λ(λ, T) = (2hc²/λ⁵) * 1/(e^(hc/λk_BT) - 1)`
        
        Parameters
        ----------
        wavelength : float or ndarray
            Longitud de onda [m]
        temperature : float or ndarray
            Temperatura [K]
            
        Returns
        -------
        B_lambda : float or ndarray
            Specific Intensity for Blackbody Radiation [W·sr⁻¹·m⁻³]
        """
        
        # Constantes físicas (SI)
        h = 6.62607015e-34      # Constante de Planck [J·s]
        c = 299792458           # Velocidad de la luz [m/s]
        k_B = 1.380649e-23      # Constante de Boltzmann [J/K]

        # Exponente
        exp_factor = np.exp((h * c) / (wavelength * k_B * temperature)) - 1.0
        
        # Función de Planck
        B_lambda = (2.0 * h * c**2) / (wavelength**5) * (1.0 / exp_factor)
        
        return B_lambda

    def blackbody_photons(wavelength, temperature):
        """
        Planck's Blackbody distribution function for Photon Flux :math:`J_λ`

        :math:`J_λ(λ, T) = πB_λ(λ, T) / (hc/λ)`

        Parameters
        ----------
        wavelength : float or ndarray
            Wavelenght [m]
        temperature : float or ndarray
            Temperature [K]

        Returns
        -------
        J_lambda : float or ndarray
            Specific Photon Flux for Blackbody Radiation [photons·sr⁻¹·m⁻²·s⁻¹·m⁻¹]
        """

        # Constantes físicas (SI)
        h = 6.62607015e-34      # Constante de Planck [J·s]
        c = 299792458           # Velocidad de la luz [m/s]
        k_B = 1.380649e-23      # Constante de Boltzmann [J/K]

        # BlackBody Distribution 
        B_lambda = Science.blackbody_intensity(wavelength, temperature)

        # BlackBody Photons Distribution
        J_lambda = mh.pi * B_lambda / (h * c / wavelength)
        
        return J_lambda

    def integrate_planck_flux(T, lambda_min, lambda_max):
        """
        Integrate Planck's blackbody distribution function over a wavelength range.

        :math:`\int_{\lambda_{min}}^{\lambda_{max}} B_λ(λ, T) dλ`
        
        Parameters
        ----------
        T : float
            Temperature of the blackbody [K]
        lambda_min : float
            Lower limit of wavelength integration [m]
        lambda_max : float
            Upper limit of wavelength integration [m]
            
        Returns
        -------
        result : float
            Integrated spectral radiance over the wavelength interval [W·sr⁻¹·m⁻²]
        """
        
        result, error = quad(Science.blackbody_intensity, lambda_min, lambda_max, args = (T,))
        
        return result
    
    def integrate_planck_photons(T, lambda_min, lambda_max):
        """
        Integrate Planck's blackbody photon distribution function over a wavelength range.

        :math:`\int_{\lambda_{min}}^{\lambda_{max}} J_λ(λ, T) dλ`

        Parameters
        ----------
        T : float
            Temperature of the blackbody [K]
        lambda_min : float
            Lower limit of wavelength integration [m]
        lambda_max : float
            Upper limit of wavelength integration [m]

        Returns
        -------
        result : float
            Integrated photon flux over the wavelength interval [photons·sr⁻¹·m⁻²·s⁻¹]
        """

        result, error = quad(Science.blackbody_photons, lambda_min, lambda_max, args = (T,))

        return result

    def get_convexhull(data):
        if len(data)>0:
            try:
                qhull=ConvexHull(data)
            except:
                qhull=None
        else:
            qhull=None
        return qhull
        
    def points_in_hull(p, hull, tol=1e-12):
        """
        Determine if a set of points are inside a convex hull.
        
        Parameters
        ---------------    
        p : `np.array(Nx2)`
            Set of coordinates for points to evaluate.
    
        hull : `scipy.spatial.ConvexHull`
            Convex hull to evaluate.
                
        Returns
        ------------
        :
            inside : `np.array(N)`
                Boolean array telling if points are inside the convex hull.

        References
        ---------------
        Method taken from https://stackoverflow.com/a/72483841
                
        Examples
        -------------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> # Random sampling points to define a ConvexHull
        >>> rng = np.random.default_rng()
        >>> points = rng.random((30, 2))
        >>> hull = ConvexHull(points)
        >>> 
        >>> # Determine which points are inside the ConvexHull
        >>> ps = rng.random((30, 2))-0.5
        >>> cond = points_in_hull(ps,hull)
        >>> 
        >>> # Plotting the Hull and points in/out it
        >>> plt.figure()
        >>> for simplex in hull.simplices:
        >>>    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        >>>
        >>> for p in ps[cond]:
        >>>    plt.plot(p[0],p[1],'r*')
        >>> 
        >>> for p in ps[~cond]:
        >>>    plt.plot(p[0],p[1],'co')

        .. image:: images/science_convexhull.png
            :align: center
            :width: 600px  
        """
        return np.all(hull.equations[:,:-1] @ p.T + np.repeat(hull.equations[:,-1][None,:], len(p), axis=0).T <= tol, 0)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class Plane
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Plane(PrynglesCommon):
    """Class with methods to construct a plane in 3-Dimensional space. 
    It is useful for geometric manipulation of points on a ring since it is considered a plane
    
    Parameters
    ---------------
    p1, p2, p3 : `np.array(N = 3)`
        3-set of points to define the plane.
            
    Attributes
    -------------
    a, b, c, d : `float`
        Coefficients of the equation of the plane

        .. math::
           a x + b y + c z + d = 0
        
    nx, ny, nz : `float`
        Cartesian components of normal vector to the plane
            
    Note
    -------------
    This class has been optimized removing all vectorial
    operations. This reduce considerably the execution time.

    Examples
    -----------
    >>> # Create a plane from a set of 3 points
    >>> p1 = [-1, 2, 1]
    >>> p2 = [0, -3, 2]
    >>> p3 = [1, 1, -4]
    >>> plane = pr.Plane(p1, p2, p3)
    >>> plane.plot_plane()
    
    .. image:: images/science_plane.png
            :align: center
    """
    def __init__(self,p1,p2,p3):
        x1,y1,z1=p1
        x2,y2,z2=p2
        x3,y3,z3=p3
        
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1

        self.a = b1 * c2 - b2 * c1
        self.b = a2 * c1 - a1 * c2
        self.c = a1 * b2 - b1 * a2
        self.d = (- self.a * x1 - self.b * y1 - self.c * z1)
                
        #Save components of the defining points
        self.p1x = p1[0];self.p1y = p1[1];self.p1z = p1[2]
        self.p2x = p2[0];self.p2y = p2[1];self.p2z = p2[2]
        self.p3x = p3[0];self.p3y = p3[1];self.p3z = p3[2]

        #Normal vector
        self.normal=(self.a**2+self.b**2+self.c**2)**0.5
        self.nx=self.a/self.normal
        self.ny=self.b/self.normal
        self.nz=self.c/self.normal
    
    def get_projection(self,p):
        """Find the projected point on the surface of the plane.
        
        Parameters
        ------------
        p : `list`
            Cartesian coordinates of the point.
        
        Returns
        ----------
        :
            v : `list`:
                Cartesian coordinates of projection point.
                
            d : `float`
                Distance from evaluated point to its projection

        Examples
        -----------
        >>> # From the previous plane constructed
        >>> p_test = [2,2,5]
        >>> v, dist = plane.get_projection(p)
        >>>
        >>> print(f'Distance: {dist}')
        >>> print(f'Projection: {v}')
        Distance: 4.015478735955178
        Projection: [-1.67741935483871, 1.0099255583126552, 3.727047146401985]

        >>> # We can also plot this point and its projection
        >>> plane.plot_plane(p = p, alpha = 0.1, color = 'r')
        
        .. image:: images/science_plane_point.png
            :align: center
        """
        
        #Distance
        d=abs(self.a*p[0]+self.b*p[1]+self.c*p[2]+self.d)/self.normal
        
        #Vectorial equivalent np.dot(p-self.p1,self.n)
        pdn=(p[0]-self.p1x)*self.nx+(p[1]-self.p1y)*self.ny+(p[2]-self.p1z)*self.nz

        #Vectorial equivalent v=p-np.dot(p-self.p1,self.n)*self.n
        v=[0]*3
        v[0]=p[0]-pdn*self.nx
        v[1]=p[1]-pdn*self.ny
        v[2]=p[2]-pdn*self.nz
        
        return v,d
    
    def get_z(self,x,y):
        """Get z value of a plane corresponding to given x, y coordinates.
        """
        if mh.isclose(self.c, 0):
            # Vertical plane: return a nan array matching x's shape (plot_surface
            # requires Z to be a 2D array, not a scalar).
            return np.full_like(x, np.nan)
        z = (-self.a*x-self.b*y-self.d)/self.c
        return z
    
    def is_above(self,p,vdir):
        """Check if a point is above or below a plane with respect to a given direction
        
        Parameters
        --------------
        p : `list`
            Cartesian coordinates of the point.
                
        vidr : `list`
            Vector direction

        Examples
        ---------
        >>> # With this method, we can evaluate if a point is or not transiting over a Ring
        >>> p = [2, 2, 5]
        >>> print(f"Is transiting: {plane.is_above(p, [0, 0, -1]    )}")
        Is transiting:  False
        """
        v,d=self.get_projection(p)        
        #Sign of (v-p).vdir
        cdir=(v[0]-p[0])*vdir[0]+(v[1]-p[1])*vdir[1]+(v[2]-p[2])*vdir[2]
        return cdir<=0
    
    def is_below(self,p,vdir):
        return not self.is_above(p,vdir)
    
    def plot_plane(self,ax=None,p=None,**args):
        
        if ax is None:
            fig=plt.figure()
            ax=fig.add_subplot(111,projection='3d')
        
        maxval=max(abs(self.p1x),abs(self.p1y),abs(self.p1z),
                   abs(self.p2x),abs(self.p2y),abs(self.p2z),
                   abs(self.p3x),abs(self.p3y),abs(self.p3z))
        
        if p is not None:
            maxval=max(maxval,abs(p[0]),abs(p[1]),abs(p[2]))

        X,Y = np.meshgrid(np.linspace(-maxval,+maxval),np.linspace(-maxval,+maxval))
        Z=self.get_z(X,Y)
        
        ax.plot_surface(X,Y,Z,**args)
        ax.plot([self.p1x,self.p2x,self.p3x],
                [self.p1y,self.p2y,self.p3y],
                [self.p1z,self.p2z,self.p3z],'co')
        
        f=2
        ax.set_xlim(-f*maxval,+f*maxval)
        ax.set_ylim(-f*maxval,+f*maxval)
        ax.set_zlim(-f*maxval,+f*maxval)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        if ax is None:
            ax.set_title(f"Plane: $a = {self.a:.2f}$, $b = {self.b:.2f}$, $c = {self.c:.2f}$, $d = {self.d:.2f}$")
        
        self.ax=ax
        self.maxval=maxval
        
        if p is not None:
            self.plot_point(p)
        
    def plot_point(self,p):
        maxval=max(self.maxval,abs(p[0]),abs(p[1]),abs(p[2]))
        ax=self.ax
        
        ax.plot(p[0],p[1],p[2],'ro')
        v,d=self.get_projection(p)
        ax.plot(v[0],v[1],v[2],'b*')

        ax.plot([p[0],v[0]],
                [p[1],v[1]],
                [p[2],v[2]],
                'r--')

        f=2
        ax.set_xlim(-f*maxval,+f*maxval)
        ax.set_ylim(-f*maxval,+f*maxval)
        ax.set_zlim(-f*maxval,+f*maxval)
