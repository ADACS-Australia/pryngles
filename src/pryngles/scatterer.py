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

from abc import ABC, abstractmethod
from scipy.optimize import bisect
from scipy.integrate import quad,dblquad
from scipy.interpolate import interp1d, RectBivariateSpline


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class Scatterer
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Scatterer(PrynglesCommon,ABC):
    """
    Abstract base class for scattering surfaces or atmospheres. 
    This class defines the interface and registration system for all scatterers
    used in photometric modeling. 

    Caution
    ------------
    Subclasses must implement the ``get_albedo()`` method,
    which computes the reflectance based on geometric parameters.

    Examples
    -----------
    >>> # You can create your own Scatterer
    >>> class MySurface(Scatterer):
    >>> 
    >>>     # Read and initialize scatterer parameters
    >>>     def __init__(self, **params): 
    >>>         if self.register(self, params):
    >>>             self.A = params["A"]
    >>>             self._initialize_scatterer()
    >>> 
    >>>     # Mandatory Method
    >>>     def get_albedo(self,eta,zeta,delta,lamb,**params):
    >>>         albedo = self.AA*eta
    >>>         return albedo
    >>> 
    >>>     # Private methods to prepare scatterer
    >>>     def _initialize_scatterer(self):
    >>>        self.AA = self.A**2
    """

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Bassic methods
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    @abstractmethod
    def __init__(self,**params)->str:
        """To read and initialize the scatterer parameters"""
        pass
    
    @abstractmethod
    def get_albedo(self,eta:float,zeta:float,delta:float,lamb:float,**params)->float:
        """Method to compute and provide the geometrical albedo of a scatterer surface"""
        pass
    
    @classmethod
    def register(self,scatterer,params):
        """Method to register a particular scatterer
        """
        scatterer.params=params
        scatterer.params["name"]=scatterer.__class__.__name__
        scatterer.hash=Misc.calc_hash(params)
        if scatterer.hash in SCATTERERS_CATALOGUE:
            verbose(VERB_SIMPLE,f"Scatterer with name {scatterer.params['name']} and hash {scatterer.hash} already exist at {id(SCATTERERS_CATALOGUE)}")
            scatterer.__dict__=deepcopy(SCATTERERS_CATALOGUE[scatterer.hash].__dict__)
            return False
        else:
            verbose(VERB_SIMPLE,f"Creating a new scatterer with name {scatterer.params['name']} and hash {scatterer.hash}")
            scatterer.params["hash"]=scatterer.hash
            SCATTERERS_CATALOGUE[scatterer.hash]=scatterer
            return True
        
    @classmethod
    def reset_catalogue(self):
        """To reset the catalogue of registered scatterers """
        # Mutate the shared global in place (register() populates the same
        # dict), otherwise this would only rebind a local name.
        SCATTERERS_CATALOGUE.clear()
        


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class NeutralSurface
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class NeutralSurface(Scatterer):
    """
    Idealized scattering surface with constant unit albedo. 
    Represents a perfectly reflecting surface, independent of geometry or wavelength.
    """
    def __init__(self,**params):
        if self.register(self,params):
            pass
    
    def get_albedo(self,eta,zeta,delta,lamb,**params):
        """Returns an albedo of 1 for all inputs"""
        return 1
    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class BlackBodySurface
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class BlackBodySurface(Scatterer):
    """
    Idealized absorbing surface with zero albedo. Represents a perfect black body that absorbs all incoming radiation.
    """
    def __init__(self,**params):
        if self.register(self,params):
            pass
    
    def get_albedo(self,eta,zeta,delta,lamb,**params):
        """Returns an albedo of 0 for all inputs"""
        return 0
    


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class LambertianGraySurface
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class LambertianGraySurface(Scatterer):
    """
    This is the scatterer corresponding to a surface having a gray lambertian Albedo with optional phase law.
    Models a surface that reflects light isotropically with a constant (gray) albedo.

    Parameters
    ----------
    AL : `float`
        Wavelength-independent albedo of the surface (0 ≤ AL ≤ 1). 
        It is interpreted as the hemispherical albedo under normal incidence.

    phase_law : function, optional
        | Law of diffuse reflection used to describe the scattering behavior of the surface and compute the angular dependence of the surface albedo.
        | By default, a Lambertian phase law is assumed:

        .. code-block:: python

            lambda eta, zeta, delta, lamb, params: eta
        
        | An alternative commonly used in planetary science is the **Lommel-Seeliger law**:

        .. code-block:: python

            lambda eta, zeta, delta, params: eta*zeta/(eta + zeta)

    
    Note
    -----------
    The phase law can be customized. This are the functional form of Lambert's cosine law & Lommel-Seeliger law.

    .. math::

        f(\eta, \zeta) = 
        \\begin{cases}
            \cos\eta, & \\text{(Lambertian phase law)} \\\\
            \dfrac{\cos\eta \cos \zeta}{\cos\eta + \cos \zeta}, & \\text{(Lommel-Seeliger phase law)}
        \end{cases}
    
    Where :math:`\eta` is the angle of incidence and :math:`\zeta` is the angle of reflection or emission.
    
    The provided `phase_law` function must follow this prototype:

    .. code-block:: python

        def phase_law(eta, zeta, delta, lamb, **params):
            '''
            Phase/Diffuse-Reflection law of the surface.

            Parameters
            ----------
            eta : float
                Cosine of the incoming angle.
            zeta : float
                Cosine of the outgoing angle.
            delta : float
                Azimuthal angle difference between incoming and outgoing directions.
            lamb : float
                Wavelength.
            params : dict
                Additional parameters required by the phase law.
            '''
    """
    
    def __init__(self,**params):

        
        if self.register(self,params):
            verbose(VERB_SIMPLE,f"Initializing {self.params['name']} with hash {self.hash}")
            
            #Phase law
            if "phase_law" in params:
                self.phase_law=params["phase_law"]
            else:
                self.phase_law=lambda eta,zeta,delta,lamb,params:eta

            #Gray albedo
            self.AL=params["AL"]

            #Calculate the gammap parameter
            self.gammap0=self._find_gammap()

            #Accelerate the calculation of the albedo
            self._accelerate_lambertian_albedo()

    def get_albedo(self,eta,zeta,delta,lamb,**params):
        """ 
        Compute the directional albedo for a given incident angle :math:`\eta` in a planetary gray Lambertian surface, assuming a gray, isotropic scattering law. 

        Parameters
        ----------
        eta : float
            Cosine of the incoming angle.
        zeta : float
            Cosine of the outgoing angle.
        delta : float
            Azimuthal angle difference between incoming and outgoing directions.
        lamb : float
            Wavelength.
        AL : `float`
            Wavelength-independent albedo of the surface (0 ≤ ``AL`` ≤ 1). 
            It is interpreted as the hemispherical albedo under normal incidence
        phase_law : function, optional
            Law of diffuse reflection used to describe the scattering behavior of the surface and compute the angular dependence of the surface albedo.

        Returns
        -------
        :
            `float`
                Wavelength-independent Lambertian directional albedo :math:`A_L(\eta)` at the given incident angle :math:`\eta`.

        Note
        ------------
        | The directional-dependent albedo is precomputed via numerical integration of the phase law and interpolated for efficiency.
        Since you provide a value for surface albedo ``AL``, 
        we implement a root method to find the `single scattering albedo` :math:`\gamma` in order to compute 
        the directional dependence (:math:`\cos\eta_i`) of albedo (Eq. 12) **[1]**, where :math:`\eta_i` refers to the 
        incidence angle of the light on each of the surface's `Spangles`.}

        .. math:: 

            A_L(\eta_i) = 2\pi\gamma\int_0^1\\frac{f(\eta_i,\,\zeta)}{\cos\eta_i}\,d(\cos\zeta)

        **[1]** Zuluaga, J. I., Sucerquia, M., & Alvarado-Montes, J. A. (2022). 
        `The bright side of the light curve: A general photometric model of non-transiting exorings`. 
        Astronomy and Computing 40 (2022) 100623. `arXiv:2207.08636 <https://arxiv.org/abs/2207.08636>`_
        """

        return self._get_albedo(eta)
        
    #####################################
    #Complimentary routines
    #####################################
    def _calc_lambertian_albedo(self,eta,gammap0=1):
        if eta==0:return self.AL
        integrand=lambda zeta:self.phase_law(eta,zeta,0,0,0)/eta
        AL=2*np.pi*gammap0*quad(integrand,0,1)[0]
        return AL

    def _find_gammap(self):
        function=lambda gammap0:self._calc_lambertian_albedo(1,gammap0)-self.AL
        gammap0=bisect(function,0.0,1.0,rtol=1e-3)
        return gammap0 if gammap0<=1 else 1
    
    def _accelerate_lambertian_albedo(self):
        etas=np.linspace(0.0,1.0,20)
        ALs=np.array([self._calc_lambertian_albedo(eta,gammap0=self.gammap0) for eta in etas])
        self._get_albedo=interp1d(etas,ALs)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Class LambertianGrayAtmosphere
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class LambertianGrayAtmosphere(Scatterer):
    """
    This is the scatterer corresponding to a semi-infinite (:math:`\\tau\\to\infty`), plane-parallel atmosphere with gray Lambertian scattering.
    Models the diffuse reflection properties assuming an atmosphere composed of particles that scatter isotropically
    
    Parameters
    ----------------
    AS : `float`
        Spherical, wavelength-independent albedo of the atmosphere (0 ≤ ``AS`` ≤ 1).
        It is interpreted as the desired hemispherical albedo under normal incidence.
    """
    
    def __init__(self,**params):

        if self.register(self,params):        
            verbose(VERB_SIMPLE,f"Initializing {self.params['name']} with hash {self.hash}")
            
            #Gray albedo
            self.AS=params["AS"]

            #Load reflection functions
            self._load_reflection_functions()

            #Calculate the gammap parameter
            self.gamma0=self._find_gamma()

            #Accelerate the calculation of the albedo
            self._accelerate_lambertian_albedo()

    def get_albedo(self,eta,zeta,delta,lamb,**params):
        """ 
        Compute the directional Lambertian albedo :math:`A_L(\eta)`, at a given incident angle of illumination :math:`\eta`, of a planetary atmosphere
        assuming gray scattering and a semi-infinite layers.  

        Parameters
        ----------
        eta : float
            Cosine of the incoming angle.
        zeta : float
            Cosine of the outgoing angle.
        delta : float
            Azimuthal angle difference between incoming and outgoing directions.
        lamb : float
            Wavelength.
        AS : `float`
            Spherical, wavelength-independent albedo of the atmosphere (0 ≤ ``AS`` ≤ 1).
            It is interpreted as the desired hemispherical albedo under normal incidence.

        Returns
        -------
        :
            `float`
                directional-dependent Lambertian albedo :math:`A_L(\eta)` of the atmosphere for a given incident angle :math:`\eta`.

        Note
        -------
        For a given spherical albedo, we derive, by root-finding methods, the  `single scattering albedo` :math:`\gamma` that reproduces the desired hemispheric reflectance ``AS`` (Eq. 10) **[2]**
        
        .. math::

            A_S = 4 \int_0^1 \int_0^1 \cos \Lambda \cos Z \, \\rho(\gamma, \Lambda, Z) \, d(\cos \Lambda) \, d(\cos Z)

        We also implement a 2D interpolation of pre-tabulated reflection coefficient :math:`\\rho(\gamma, \eta, \zeta)` (Eq. 7) **[2]**, 
        based on radiative transfer solutions (Table 2.3  in Sobolev, 1975) **[1]** to model the direction-dependent Lambertian albedo efficiently (Eq. 8) **[2]**.

        .. math::

            \\rho(\gamma, \Lambda, Z) = \\frac{\gamma}{4} \\frac{f(\gamma, Z) \, f(\gamma, \Lambda)}{\cos \Lambda + \cos Z}

        .. math::

            A_{L_i}(\Lambda_i) = 2 \int_0^1 \cos Z \, \\rho(\gamma, \Lambda_i, Z) \, d(\cos Z)


        References
        -----------------------
        **[1]** Sobolev, V. V. (1975). Light Scattering in Planetary Atmospheres. 
        
        **[2]** Zuluaga, J. I., Sucerquia, M., & Alvarado-Montes, J. A. (2022). 
        `The bright side of the light curve: A general photometric model of non-transiting exorings`. 
        Astronomy and Computing 40 (2022) 100623. `arXiv:2207.08636 <https://arxiv.org/abs/2207.08636>`_
        """
        return self._get_albedo(eta)
        
    #####################################
    #Complimentary routines
    #####################################
    def _load_reflection_functions(self):
        """Load value of reflection fucntions.

        Update:
            fint: 2d interpolating function:
                x: eta (cosine incident angle)
                y: zeta (cosine scattering angle)

        Notes:
            Tab. (2.3) in Sobolev (1975).
        """
        data_ss=np.loadtxt(Misc.get_data("diffuse_reflection_function.data"))
        eta=data_ss[1:,0]
        gamma=data_ss[0,1:]
        f=data_ss[1:,1:]
        # SciPy >= 1.14 removed interp2d. We use a regular-grid spline instead.
        # interp2d(x=gamma, y=eta, z=f) expects z.shape == (len(eta), len(gamma))
        # and is called as fint(gamma0, eta0) -> array([[value]]).
        # (mirrors the fix already applied in legacy.py).
        spline = RectBivariateSpline(eta, gamma, f, kx=1, ky=1)

        def fint(g, e):
            """
            Backward-compatible replacement for interp2d(gamma, eta, f).

            interp2d(x, y, z) returns:
              - (x scalar, y scalar) -> array([z])
              - (x scalar, y vector) -> array([z(y_i)])
              - (x vector, y scalar) -> array([z(x_i)])
              - (x vector, y vector) -> 2D array with shape (len(y), len(x))
            """
            gg = np.atleast_1d(g).astype(float)
            ee = np.atleast_1d(e).astype(float)

            if gg.size == 1 and ee.size == 1:
                return np.array([float(spline(ee[0], gg[0])[0, 0])])

            if gg.size == 1 and ee.size > 1:
                return spline(ee, gg).reshape(-1)

            if gg.size > 1 and ee.size == 1:
                return spline(ee, gg).reshape(-1)

            return spline(ee, gg)

        self.fint = fint

    def _calc_reflection_coefficient(self,eta,zeta,gamma0=1):
        """Reflection coefficient of a semi-infinite (tau = infinity) atmosphere with (gray)
        single scattering albedo gamma0

        Requires:
            - _loadReflectionFunctions

        Notes:
            Ec. (2.43) in Sobolev (1975).
        """
        rho0=gamma0*self.fint(gamma0,eta)[0]*self.fint(gamma0,zeta)[0]/(4*(eta+zeta))
        return rho0

    def _calc_spherical_albedo(self,gamma0):
        """
        Compute spherical albedo from single scattering albedo for a semi-infinite atmosphere.

        Parameters:
            gamma0: single scattering albedo (0<=gamma0<=1), float.

        Returns:
            AS: ratio of the energy diffusely reflected by a spherical planet (0<=AS<=1), float.

        Requires:
            - _loadReflectionFunctions

        Notes:
            Ec. (1.87) in Sobolev (1975).    
        """

        AS=4*dblquad(lambda y,x,*args:self._calc_reflection_coefficient(x,y,*args)*x*y,
                     0,1,lambda x:0,lambda x:1,epsrel=1e-2,args=(gamma0,))[0]
        return AS

    def _find_gamma(self):
        """
        Starting with a target spherical albedo AS, find the value of the single scattering albedo gamma0
        of a semi-infinite atmosphere having that Albedo.

        Returns:
            gamma0: the value of gamma0 corresponding to AS (0<=gamma0<=1), float.
        """
        if np.isclose(self.AS,1,rtol=1e-2):
            return 1
        function=lambda gamma0:self._calc_spherical_albedo(gamma0)-self.AS
        gamma0=bisect(function,0.0,1.0,rtol=1e-4)
        return gamma0 if gamma0<=1 else 1

    def _calc_lambertian_albedo(self,eta):
        """
        Notes: 
            Yanovistkii (1973)
        """
        integrand=lambda zeta:self._calc_reflection_coefficient(eta,zeta,gamma0=self.gamma0)*zeta
        AL=2*quad(integrand,0,1,epsrel=1e-3)[0]
        return AL

    def _accelerate_lambertian_albedo(self):
        etas=np.linspace(0,1,20)
        ALs=np.array([self._calc_lambertian_albedo(eta) for eta in etas])
        self._get_albedo=interp1d(etas,ALs)
