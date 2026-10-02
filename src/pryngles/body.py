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

import math

import numpy as np
from anytree import RenderTree
from scipy.interpolate import interp1d

from pryngles import science
from pryngles.common import VERB_VERIFY, PrynglesCommon, verbose
from pryngles.consts import (
    DETECTOR_PROPERTIES,
    OBSERVER_DEFAULTS,
    PLANET_DEFAULTS,
    REBOUND_ORBITAL_PROPERTIES,
    RING_DEFAULTS,
    SCIENCE_LIMB_NORMALIZATIONS,
    STAR_DEFAULTS,
    T_MODEL_DEFAULTS,
    Consts,
)
from pryngles.extensions import StokesScatterer
from pryngles.misc import get_data
from pryngles.orbit import Orbody
from pryngles.spangler import Spangler

# --------------------------------------------------
# Class Body
# --------------------------------------------------


class Body(Orbody):
    """
    This is the class to create a general body in `pryngles`.

    Note
    -----
    This class is not intended to be used independently, just for inheritance purposes.

    Parameters
    ----------
    kind : `str`
        One of the kind of bodies defined in the package (:any:`consts.BODY_KINDS`)

    defaults : `OrderedDict`, `dict`
        Dictionary with the properties of the object (:any:`consts.BODY_DEFAULTS`)

    parent: :data:`~ body.Body`
        Object in the center of the orbit of this body.

    **properties: `dict`
        Specify additional body properties and its values from :any:`consts.REBOUND_ORBITAL_PROPERTIES`
        or :any:`consts.BODY_DEFAULTS`

    Returns
    -------
    :
        output : :data:`~ body.Body`
            Body object containing the physical, orbital and optical parameters for an astropyshical body

    Raises
    ------
    AssertionError
        If **parent** parameter is not a valid :data:`~ body.Body` object

    Attributes
    ----------
    sg : :any:`spangler.Spangler`
        Abbreviation of `spangler`. This is one of the most important objects in ``pryngles``.
        It contains the :data:`~ spangler.Spangler` object in wich we sample and discretize the surface
        of the :data:`~ body.Body` object in order to compute light-matter interactions.
        | **Default** is ``None``.

    childs, children: `dict, tuple`
        It contains child bodies (bodies which is having this body) as the center.

    See Also
    -------------
    :any:`spangler.Spangler`
        Visit our :doc:`spangler` reference to detailed explanation of the class and its purposes.

    Examples
    --------
    This brief example shows how to create a bodies system with a primary body

    >>> # Create a body with None parent and name = 'B'
    >>> B = pr.Body(kind = "Body", defaults = pr.BODY_DEFAULTS, parent = None, name='B')
    >>> # Create a body having parent the Body "B" defined before:
    >>> C = pr.Body(kind = "Body", defaults = BODY_DEFAULTS, parent = B, name="C")
    """

    # --------------------------------------------------
    # Bassic methods
    # --------------------------------------------------

    def __init__(self, kind, defaults, parent, **props):

        # Kind, parent and child attributes
        self.kind = kind
        self.__defaults = defaults

        # Prepare key attributes
        self.sg = None

        # Name of the object
        if "name" in props:
            name = self.name = str(props["name"])
        elif "name_by_kind" in props:
            name = self.name = self.kind
        else:
            name = self.name = str(hash(self))

        # Legacy
        if "primary" in props:
            parent = props["primary"]
        if "optics" in props:
            props.update(props["optics"])
        if "orbit" in props:
            props.update(props["orbit"])
        if "physics" in props:
            props.update(props["physics"])

        # Update childs and parent
        if parent is not None:
            if not isinstance(parent, Body):
                raise AssertionError(f"Parent is not a valid Object: {type(parent)}, {isinstance(parent, Body)}")
            else:
                self.parent = parent
                parent._update_childs(self)

        # Update parent and childs
        self._update_parent(parent)
        self._update_childs()

        # Update default properties
        self.__dict__.update(defaults)
        # Set name
        self.name = name
        # Update body
        self.update_body(**props)

    def update_body(self, **props):
        """
        Update properties of the Body.

        Parameters
        ----------------------
        **props: dictionary:
            Properties to update.
            The current object is updated with new values provided in this new object

        Examples
        ---------------
        >>> # Let's create a Body first
        >>> B = pr.Body(kind = "Body", defaults = pr.BODY_DEFAULTS, parent = None, name='B')
        >>> # This only update the Body mass attribute.
        >>> B.update_body(m=2)
        """
        for prop in props:
            if prop in self.__defaults or prop in REBOUND_ORBITAL_PROPERTIES:
                self.__dict__[prop] = props[prop]
            else:
                raise ValueError(f"Property {prop} not identified in object {self.kind}")

        self.elements = {k: v for k, v in self.__dict__.items() if k in REBOUND_ORBITAL_PROPERTIES}

        verbose(VERB_VERIFY, "Updating Body")
        self._update_properties()

    def _update_childs(self, child=None):
        if "childs" not in self.__dict__:
            self.childs = dict()
        if child is not None:
            verbose(VERB_VERIFY, f"Add child {child.name} to body {self.kind} ({self.name})")
            self.childs[child.name] = child

    def _update_parent(self, parent=None):
        if "parent" not in self.__dict__:
            if parent:
                verbose(VERB_VERIFY, f"Add parent {parent.name} to body {self.kind} ({self.name})")
            self.parent = parent
        elif parent is not None:
            verbose(VERB_VERIFY, f"Add parent {parent.name} to body {self.kind} ({self.name})")
            self.parent = parent
            parent._update_childs(self)

    def _update_properties(self):
        verbose(VERB_VERIFY, "Updating properties of Body")
        # Rotational angular velocity
        self.wrot = 2 * np.pi / self.prot
        # Rotation axis
        self.n_equ = science.cartesian([1, self.roll, 90 * Consts.deg - self.i])

    def show_tree(self):
        print(RenderTree(self))

    # --------------------------------------------------
    # Tested methods from module file body
    # --------------------------------------------------

    def spangle_body(self):
        """
        Spangle the surface of the body. It creates and updates the :data:`~ spangler.Spangler` object in wich we
        generate a sampling for the discrete units (`Spangles`) over the whole area on the body

        Examples
        --------
        >>> # Once we Spangle the Body, we can access to its most importante atribute
        >>> B.spangle_body()
        >>> # The data attribute is a pandas DataFrame object
        >>> # It contains all the geometrical and state data that caracterize the body surface
        >>> B.sg.data
            name  spangle_type  geometry  scale                              n_equ  alpha_equ  ...   emit  above illuminated  transmit  transit  occult
        0      B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        1      B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        2      B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        3      B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        4      B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        ..   ...           ...       ...    ...                                ...        ...  ...    ...    ...         ...       ...      ...     ...
        982    B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        983    B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        984    B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        985    B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False
        986    B             6         1    0.1  [6.123233995736766e-17, 0.0, 1.0]          0  ...  False  False        True     False    False   False

        See Also
        --------------
        :any:`consts.SPANGLER_COLUMNS`
            To see a description for each column in the data attribute
        """  # noqa: E501

        # Create spangler
        self.sg = Spangler(
            nspangles=self.nspangles,
            name=self.name,
            n_equ=self.n_equ,
            alpha_equ=self.alpha,
            w=self.wrot,
            q0=self.q0,
            center_equ=self.center_equ,
            center_ecl=self.center_ecl,
        )

        # Populate spangler
        self.sg.populate_spangler(
            shape=self.shape,
            spangle_type=self.spangle_type,
            scale=self.radius,
            seed=self.seed,
            preset=self.preset,
            **self.geometry_args,
        )

        self.sg.set_positions()

        # Additional properties in the Spangler DataFrame
        if self.kind == "Star":
            self.sg.data.source = True

        self.sg.set_observer()
        self.sg.set_luz()


# --------------------------------------------------
# Class Star
# --------------------------------------------------
class Star(Body):
    """It creates a :data:`~ body.Star` object for self-luminous objects with emission properties.

    Parameters
    --------------------
    parent: :data:`~ body.Star`, `None`
        Body for wich Star is orbiting.
        For ``None`` it means Star is the main body in the system
        | **Default** is ``None``.

    **props: `dict`
        Specify additional Star properties and its values.
        For the complete set of default values of the properties see :any:`consts.STAR_DEFAULTS`

    Returns
    -------
    :
        output : :data:`~ body.Star`
            Star body object containing the physical, orbital and optical parameters.

    Raises
    ------
    ValueError
        If **parent** parameter is not a valid :data:`~ body.Star` object.
        Only another :data:`~ body.Star` can be the parent of a :data:`~ body.Star`
    """

    def __init__(self, parent=None, **props):

        # Instantiate object with basic properties
        Body.__init__(self, "Star", STAR_DEFAULTS, parent, **props)

        # Check parent
        if self.parent is not None:
            if self.parent.kind != "Star":
                raise ValueError(f"Only another Star can be the parent of a Star (you provided {self.parent.kind})")

        self._update_star_properties()

    def _update_star_properties(self):

        verbose(VERB_VERIFY, "Updating properties of Star")

        # Compute limbdarkening at r = 0 to initialize normalization constant
        science.limb_darkening(0, self.limb_coeffs)

        # Store limb darkening normalization
        self.norm_limb_darkening = SCIENCE_LIMB_NORMALIZATIONS[hash(tuple(self.limb_coeffs))]

    def update_star(self, **props):
        """General and specific update properties of the Star

        Parameters
        ----------------------
        **props: dictionary:
            Properties to update.
            The current object is updated with new values provided in this new object

        Attributes
        -------------------------
        limb_coeffs: `list, array`
            Limb darkening coefficients [2]. Its lenght defines the model to implement [1].

        norm_limb_darkening: `float`
            Limb darkening function normalization.

        References
        ---------------
        [1] Models for Limb-Darkening: https://pages.jh.edu/~dsing3/David_Sing/Limb_Darkening.html
        [2] Coefficients available at: https://pages.jh.edu/~dsing3/LDfiles/LDCs.CoRot.Table1.txt
        """
        verbose(VERB_VERIFY, "Updating star")

        Body.update_body(self, **props)
        self._update_star_properties()


# --------------------------------------------------
# Class Planet
# --------------------------------------------------
class Planet(Body):
    """
    It creates a :data:`~ body.Planet` object for non-luminous orbiting bodies with specific orbital
    parameters and surface characteristics (in ``pryngles``, `Moons` are constructed as planets)

    Parameters
    --------------------
    parent: :data:`~ body.Star`
        Body for wich Planet is orbiting.

    **props: `dict`
        Specify additional Planet properties and its values.
        For the complete set of default values of the properties see :any:`consts.PLANET_DEFAULTS`

    Returns
    -------
    :
        output : :data:`~ body.Planet`
            Planet body object containing the physical, orbital and optical parameters

    Raises
    ------
    ValueError
        If `parent =  None`. Parent not provided and it is mandatory for :data:`~ body.Planet` object.
    """

    def __init__(self, parent=None, **props):

        # Instantiate object with basic properties
        Body.__init__(self, "Planet", PLANET_DEFAULTS, parent, **props)

        # Check parent
        if self.parent is None:
            raise ValueError(f"Parent not provided and it is mandatory for {self.kind}.")

        # Update properties
        self.update_planet(**props)

        # Initialize Stokes Scatterer
        self.Stokes = StokesScatterer(get_data(self.physics["fourier_file"]))

    def _update_planet_properties(self):
        verbose(VERB_VERIFY, "Updating Planet properties")

    def update_planet(self, **pars):
        """General and specific update properties of the `Planet`

        Parameters
        ----------------------
        **props: dictionary
            Properties to update.
            The current object is updated with new values provided in this new object
        """

        verbose(VERB_VERIFY, "Updating Planet")
        Body.update_body(self, **pars)
        self._update_planet_properties()

    def set_temperature_model(self, model: dict):
        """
        Set the temperature model for the planet. It defines how the temperature is distributed
        over the planet's surface.

        Parameters
        ----------
        model : `dict`
            A dictionary specifying the temperature model type and its parameters.
            The dictionary should have the following structure:
            {
                "type": "Model Type",
                "params": {
                    "param1": value1,
                    "param2": value2,
                    ...
                }
            }

        Raises
        ------
        TypeError
            If the `model` parameter is not a dictionary.
        ValueError
            If the specified model type is unknown or if mandatory parameters are missing.

        Attributes
        ----------
        T_model : `dict`
            The temperature model set for the planet, including its type and parameters.

        Examples
        --------
        >>> # Set a Uniform Temperature model with T_planet = 1500 K
        >>> planet.set_temperature_model({"type": "Uniform Temperature", "params": {"T_planet": 1500}})
        """

        if not isinstance(model, dict):
            raise TypeError(
                "Temperature model must be provided as a dictionary. It includes the model type and its parameters."
            )

        if model["type"] not in T_MODEL_DEFAULTS:
            known_models = ", ".join(T_MODEL_DEFAULTS.keys())
            raise ValueError(f"Unknown '{model['type']}' Model. Available Temperature Models are: [{known_models}]")

        model_params = model.get("params", {})

        missing_params = [k for k in T_MODEL_DEFAULTS[model["type"]].get("required", []) if k not in model_params]
        if missing_params:
            raise ValueError(f"Missing mandatory parameters for Temperature Model '{model['type']}': {missing_params}")

        self.T_model = {"type": model["type"], "params": model_params}

    def update_temperature(self):
        """ """

        if self.sg is None:
            raise RuntimeError(
                "Spangler not defined. Please spangle the body before applying a temperature model: "
                "Planet.spangle_body() or System.spangle_system()"
            )

        if not hasattr(self, "T_model"):
            raise RuntimeError(
                "Temperature model not defined. Please set a temperature model before applying it: "
                "Planet.set_temperature_model()"
            )

        model_type = self.T_model["type"]

        if model_type == "Uniform Temperature":
            self._T_uniform_temp()
        elif model_type == "Two Temperature":
            self._T_two_temp()
        elif model_type == "Zhang-Showman":
            self._T_zhang_showman()

    def _T_uniform_temp(self):

        T_planet = self.T_model["params"]["T_planet"]

        self.sg.data.Tem = T_planet

        # return self.sg.data.Tem.values

    def _T_two_temp(self):

        T_day = self.T_model["params"]["T_day"]
        T_night = self.T_model["params"]["T_night"]

        cond_day = self.sg.data.cos_luz > 0
        cond_night = ~cond_day

        self.sg.data.loc[cond_day, "Tem"] = T_day
        self.sg.data.loc[cond_night, "Tem"] = T_night

        # return self.sg.data.Tem.values

    def _T_zhang_showman(self):

        # Auxiliary
        PI = math.pi
        EXP = np.exp
        COS = np.cos

        # Barycentric Positions
        center_source = self.root.center_ecl
        center_body = self.center_ecl

        # Light Source Direction
        vec_luz = center_source - center_body
        d_luz, lamda_luz, phi_luz = science.spherical(vec_luz)

        # Longitude, Latitude
        lamda = self.sg.data["q_equ"].values - lamda_luz + PI / 2  # Long
        phi = self.sg.data["f_equ"].values - phi_luz  # Lat

        # Adjust ranges for angles
        lamda = (lamda + np.pi) % (2 * np.pi) - np.pi  # -> [-π, π]
        phi = np.clip(phi, -np.pi / 2, np.pi / 2)  # -> [-π/2, π/2]

        # Zhang & Showman Model Parameters
        T_night = self.T_model["params"]["T_night"]
        Delta_T = self.T_model["params"]["Delta_T"]
        xi_ratio = self.T_model["params"]["xi_ratio"]

        lamda_s = math.atan(xi_ratio)
        eta = (
            (xi_ratio / (1 + xi_ratio**2))
            * (EXP(PI / (2 * xi_ratio)) + EXP(3 * PI / (2 * xi_ratio)))
            / (EXP(2 * PI / xi_ratio) - 1)
        )

        # -----------------------------------------------------------------------------------

        # Temperature Model
        Ts = np.zeros_like(lamda)

        # Exponential Approximation for Xi ~ 0
        if abs(xi_ratio) < 0.01:
            EXP_1term = np.zeros_like(lamda)
            EXP_2term = np.zeros_like(lamda)
            EXP_3term = np.zeros_like(lamda)
        else:
            EXP_1term = EXP(-lamda / xi_ratio)
            EXP_2term = EXP(-(PI + lamda) / xi_ratio)
            EXP_3term = EXP((PI - lamda) / xi_ratio)

        # Región 1:  -π/2 ≤ λ ≤ π/2
        mask1 = (lamda >= -PI / 2) & (lamda <= PI / 2)
        Ts[mask1] = (
            T_night
            + Delta_T * COS(phi[mask1]) * COS(lamda_s) * COS(lamda[mask1] - lamda_s)
            + eta * Delta_T * COS(phi[mask1]) * EXP_1term[mask1]
        )

        # Región 2:  -π ≤ λ < -π/2
        mask2 = (lamda >= -PI) & (lamda < -PI / 2)
        Ts[mask2] = T_night + eta * Delta_T * COS(phi[mask2]) * EXP_2term[mask2]

        # Región 3:  π/2 < λ ≤ π
        mask3 = (lamda > PI / 2) & (lamda <= PI)
        Ts[mask3] = T_night + eta * Delta_T * COS(phi[mask3]) * EXP_3term[mask3]

        self.sg.data["Tem"] = Ts

        # return Ts


# --------------------------------------------------
# Class Ring
# --------------------------------------------------
class Ring(Body):
    """
    It creates a :data:`~ body.Ring` object for particulate systems with unique optical and physical ring properties

    Parameters
    --------------------
    parent: :data:`~ body.Star`, :data:`~ body.Planet`
        Body for wich Ring was formed around. If parent is Star type, it means a circumplanetary disk

    **props: `dict`
        Specify additional Ring properties and its values.
        For the complete set of default values of the properties see :any:`consts.RING_DEFAULTS`

    Returns
    -------
    :
        output : :data:`~ body.Ring`
            Ring body object containing the physical, orbital and optical parameters

    Raises
    ------
    ValueError
        If `parent =  None`. Parent not provided and it is mandatory for :data:`~ body.Ring` object.
    """

    def __init__(self, parent=None, **props):

        # Instantiate object with basic properties
        Body.__init__(self, "Ring", RING_DEFAULTS, parent, **props)

        # Check parent
        if self.parent is None:
            raise ValueError(f"Parent not provided and mandatory for {self.kind}.")

        # Update properties
        self.update_ring(**props)

        # Initialize Stokes Scatterer
        self.Stokes = StokesScatterer(get_data(self.physics["fourier_file"]))

    def _update_ring_properties(self):
        verbose(VERB_VERIFY, "Updating Ring properties")

        # Update radius
        self.ri = self.fi * self.parent.radius
        self.re = self.fe * self.parent.radius
        self.radius = self.re

        # Update geometry args for spangling purposes
        self.geometry_args = dict(ri=self.ri / self.re)

    def update_ring(self, **props):
        """General and specific update properties of the `Planet`

        Parameters
        ----------------------
        **props: dictionary:
            Properties to update.
            The current object is updated with new values provided in this new object

        Attributes
        -------------------------
        ri, re: `float`
            Radius of the inner (outer) border of the ring in units of the parent radius.

        albedo_gray_normal: `float`
            Lambertian (normal) gray (wavelength indpendent) albedo of the spangle.
            It takes 0 to 1 values. 1 for total reflection. Default = 1

        tau_gray_optical: `float`
            Gray (wavelength indpendent) Optical Depth of the spangle.
            Default = 0, i.e., the spangle is entirely transparent to all wavelength, despite its type.
        """
        verbose(VERB_VERIFY, "Updating Ring")
        Body.update_body(self, **props)
        self._update_ring_properties()


# --------------------------------------------------
# Class Observer
# --------------------------------------------------
class Observer(Body):
    """
    It initializes an Observer object with default properties defined in :any:`consts.OBSERVER_DEFAULTS`

    Note
    --------------
    This class is intended only for legacy purposes.

    Attributes
    -------------------------
    lamb, beta: `float`
        Ecliptic longitude/latitude of the observer in radians. Defaults to 0.
    """

    def __init__(self, parent=None, **props):
        Body.__init__(self, "Observer", OBSERVER_DEFAULTS, parent, **props)


# --------------------------------------------------
# Class Detector
# --------------------------------------------------
class Detector(PrynglesCommon):
    """
    It initializes an Detector object with default properties defined in :any:`consts.DETECTOR_PROPERTIES`.

    Note
    --------------
    Tipically, it belongs to a `~ system.System` object

    Important
    -------------
    Be sure the units are consistent with SI units when providing the properties

    Attributes
    -------------------------
    wavelength_min: `float`
        Minimum wavelength of the detector's sensitivity range in nanometers. Defaults to 500 nm.
    wavelength_max: `float`
        Maximum wavelength of the detector's sensitivity range in nanometers. Defaults to 700 nm.
    apperture: `float`
        Aperture size of the detector in meters. Defaults to 0.5 m.
    quantum_eff: `float`
        Quantum efficiency of the detector, which defines the ratio of detected photons to incident photons.
        Defaults to 1.
    t_cad: `float`
        Time of Cadence of the detector in seconds. Defaults to 10 minutes (600 s).
    distance: `float`
        Distance from the observer to the system being observed in meters [m]. Defaults to 1 kilop
    """

    def __init__(self, **props):

        # Initialize with defaults
        self.__dict__.update(DETECTOR_PROPERTIES)

        # Update with provided properties
        for prop in props:
            if prop in DETECTOR_PROPERTIES:
                self.__dict__[prop] = props[prop]
            else:
                raise AssertionError(
                    f"Invalid parameters. Valid parameters are: {', '.join(list(DETECTOR_PROPERTIES.keys()))}"
                )

    def _gaussianQuadrature(self, func, a, b):
        """
        Compute a Gaussian quadrature integral over a given interval.

        Parameters
        ----------
        func : callable
            The function to integrate. It should take a single argument.
        a : float
            The lower limit of integration.
        b : float
            The upper limit of integration.

        Returns
        -------
        float
            The approximate value of the integral of `func` from `a` to `b`.
        """

        x1 = 0
        x2 = +((3.0 / 5) ** 0.5)
        x3 = -x2
        w1 = 8.0 / 9
        w2 = w3 = 5.0 / 9
        bam2 = (b - a) / 2
        bap2 = (b + a) / 2

        try:
            integral = bam2 * (w1 * func(bam2 * x1 + bap2) + w2 * func(bam2 * x2 + bap2) + w3 * func(bam2 * x3 + bap2))
        except Exception:
            integral = bam2 * (w1 * func(bam2 * x1 + bap2) + w2 * func(bam2 * x2 + bap2) + w3 * func(bam2 * x3 + bap2))

        return integral

    def set_source(self, source: Star):
        """
        Set the source star to observe for the detector and compute the typical flux.

        Parameters
        ----------
        source : :data:`~ body.Star`
            The source star object.

        Raises
        ------
        ValueError
            If the provided source is not a :data:`~ body.Star` object.

        Attributes
        ----------
        normal_flux : `float`
            The normal flux received from the source star by the detector [photons/s].

        Notes
        -----
        Be sure the units are consistent with SI units when providing the source star
        """

        if source.kind != "Star":
            raise ValueError(f"Source must be a Star object. You provided a {source.kind} object.")

        # Compute the Source Flux
        L_star = science.integrate_planck_photons(source.T_eff, self.wavelength_min, self.wavelength_max)

        # Compute the observed Flux
        self.normal_flux = (
            self.quantum_eff
            * L_star
            * (source.radius * source.ul / self.distance) ** 2
            * (np.pi * (self.apperture / 2) ** 2)
        )

    def generate_signal(self, times, fluxes):
        """
        Generate a simulated signal from the detector.

        Parameters
        ----------
        times : array-like
            Array of time [s] points at which the fluxes are provided.
        fluxes : array-like
            Array of flux values corresponding to the time points.

        Returns
        -------
        signal_times : `numpy.ndarray`
            Times of the simulated observations [s].
        signal_lc : `numpy.ndarray`
            Simulated signal light curve [normalized flux] sampled at `signal_times`.
        signal_error : `numpy.ndarray`
            Simulated signal error [normalized flux].

        Attributes
        ----------
        times : `numpy.ndarray`
            Times of the simulated observations [s].
        signal_flux : `numpy.ndarray`
            Simulated signal flux [normalized flux].
        signal_error : `numpy.ndarray`
            Simulated signal error [normalized flux].

        Notes
        -----
        The method simulates the detection of photons by the detector over specified time intervals, in seconds [s],
        taking into account the cadence of the detector and the Poisson nature of photon detection.
        The output signal is normalized by the normal flux received from the source star.

        Examples
        --------
        >>> # Assuming  `times` and `fluxes` are defined
        >>>
        >>> detector = pr.Detector(t_cadence=600, quantum_eff=1, apperture=0.5, distance=1e3*pr.consts.Consts.pc)
        >>>
        >>> # Set the source star
        >>> star = pr.Star(T_eff = 5778, radius = 1*pr.consts.Consts.R_sun)
        >>> detector.set_source(star)
        >>>
        >>> # Generate the signal
        >>> signal_times, signal_lc, signal_error = detector.generate_signal(times, fluxes)

        .. image:: images/detector_signal_example.png
            :align: center
        """

        #########################################
        # LIGHTCURVE DATA
        #########################################
        signalFunc = interp1d(times, fluxes * self.normal_flux, kind="linear", fill_value="extrapolate")

        #########################################
        # OBSERVATIONS
        #########################################
        tmin = times[0]
        tmax = times[-1]
        t_cad = self.t_cadence  # Time of Cadence
        N_folds = 5  # Number of Folds
        N_obs = int(math.ceil((tmax - tmin) / t_cad))  # Number of Observations

        #########################################
        # SIGNAL LIGHTCURVE
        #########################################
        signal_lc = np.zeros(N_obs * N_folds)
        signal_error = np.zeros(N_folds * N_obs)
        signal_times = np.zeros(N_folds * N_obs)

        for Nf in range(N_folds):
            t_start = tmin - t_cad * np.random.rand()
            times = np.arange(t_start, t_start + t_cad * N_obs, t_cad)

            for i, t in enumerate(times):
                # PHOTONS DETECTED IN OBSERVATION TIME
                Flux_obs = self._gaussianQuadrature(signalFunc, t, t + t_cad)

                # POISSON PHOTON STOCASTIC DETECTION
                Flux_obs = np.random.poisson(Flux_obs)
                delta_Flux = math.sqrt(Flux_obs)

                signal_times[Nf * N_obs + i] = t + t_cad / 2
                signal_lc[Nf * N_obs + i] = Flux_obs / (self.normal_flux * self.t_cadence)
                signal_error[Nf * N_obs + i] = delta_Flux / (self.normal_flux * self.t_cadence)

        sort_mask = np.argsort(signal_times)

        signal_times = signal_times[sort_mask]
        signal_lc = signal_lc[sort_mask]
        signal_error = signal_error[sort_mask]

        self.times = signal_times
        self.signal_flux = signal_lc
        self.signal_error = signal_error

        return signal_times, signal_lc, signal_error
