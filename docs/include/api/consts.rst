Constants Module
========================

| This is the module for all Constants defined in ``pryngles``. 
| You will find two types of constants distinguished between `physical` and `numerical`

Physical
-----------------------

Values associated to physical and astronomical quantities, defined in **SI** units. Most of them are imported from ``rebound`` package.

.. table::
   :align: center

   +----------------+---------------------+------------+--------------------------------------------------------+
   | Attribute      | Value               | Unit       | Description                                            |
   +================+=====================+============+========================================================+
   | s              | 1.0                 | s          | Second (SI base unit)                                  |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | hr             | 3600.0              | s          | Hour                                                   |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | d              | 86400.0             | s          | Day                                                    |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | day            | 86400.0             | s          | Synonym for day                                        |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | days           | 86400.0             | s          | Plural of day                                          |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | yr             | 31557600.0          | s          | Julian year (exact)                                    |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | year           | 31557600.0          | s          | Synonym for Julian year                                |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | years          | 31557600.0          | s          | Plural of Julian year                                  |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | yrs            | 31557600.0          | s          | Abbreviation for years                                 |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | jyr            | 31557600.0          | s          | Jovian year (alias)                                    |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | sidereal_yr    | 31558149.7635       | s          | Sidereal year                                          |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | kyr            | 3.15576e10          | s          | Thousand years                                         |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | myr            | 3.15576e13          | s          | Million years                                          |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | gyr            | 3.15576e16          | s          | Billion years                                          |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | yr2pi          | 5022642.89091303    | s          | Alias for year/2π                                      |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | m              | 1.0                 | m          | Meter (SI base unit)                                   |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | cm             | 0.01                | m          | Centimeter                                             |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | km             | 1000.0              | m          | Kilometer                                              |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | au             | 149597870700.0      | m          | Astronomical Unit                                      |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | aus            | 149597870700.0      | m          | Plural of AU                                           |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | pc             | 3.085677581e16      | m          | Parsec                                                 |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | parsec         | 3.085677581e16      | m          | Synonym for parsec                                     |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | rearth         | 6.378137e6          | m          | Earth's mean radius                                    |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | rsun           | 6.957e8             | m          | Solar radius (nominal)                                 |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | rjupiter       | 7.1492e7            | m          | Jupiter's equatorial radius                            |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | rsaturn        | 6.0268e7            | m          | Saturn's equatorial radius                             |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | kg             | 1.0                 | kg         | Kilogram (SI base unit)                                |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | g              | 1e-3                | kg         | Gram                                                   |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | gram           | 1e-3                | kg         | Synonym for gram                                       |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | msun           | 1.988409870698051e30| kg         | Solar mass                                             |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | msolar         | 1.988409870698051e30| kg         | Synonym for solar mass                                 |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | msaturn        | 5.683e26            | kg         | Saturn's mass                                          |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | mjupiter       | 1.898e27            | kg         | Jupiter's mass                                         |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | mearth         | 5.972e24            | kg         | Earth's mass                                           |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | mmars          | 6.417e23            | kg         | Mars' mass                                             |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | muranus        | 8.681e25            | kg         | Uranus' mass                                           |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | mneptune       | 1.024e26            | kg         | Neptune's mass                                         |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | mmercury       | 3.302e23            | kg         | Mercury's mass                                         |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | mvenus         | 4.867e24            | kg         | Venus' mass                                            |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | mpluto         | 1.309e22            | kg         | Pluto's mass                                           |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | sunmass        | 1.988409870698051e30| kg         | Synonym for solar mass                                 |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | solarmass      | 1.988409870698051e30| kg         | Synonym for solar mass                                 |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | massist        | 2.258e25            | kg         | Normalized Earth mass unit (G=1 system)                |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | ppm            | 1e6                 | —          | Parts per million                                      |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | ppb            | 1e9                 | —          | Parts per billion                                      |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | rad            | 57.2957795          | rad/deg    | Radians per degree                                     |
   +----------------+---------------------+------------+--------------------------------------------------------+
   | deg            | 0.01745329          | deg/rad    | Degrees per radian                                     |
   +----------------+---------------------+------------+--------------------------------------------------------+


.. autoclass:: pryngles.consts.Consts
   :members:
   :member-order: bysource


Numerical
-----------------------
   
Quantities used accross the package. Usually, numerical constants are grouped by modules

Orbital Constants
^^^^^^^^^^^^^^^^^^

.. autodata:: pryngles.consts.REBOUND_ORBITAL_PROPERTIES
   :no-value:


Sampler Constants
^^^^^^^^^^^^^^^^^^^^

These are the numerical attributes to define the body shapes we consider in ``pryngles`` to Spangle and Sampling its surface. 
For `Stars`, `Planets` and `Moons` it defines a sphere-sampling. `Ring` is considered as a circle-sampling with a circle-sampling hole

.. autodata:: pryngles.consts.SAMPLE_SHAPES

.. autodata:: pryngles.consts.SAMPLER_GEOMETRY_SPHERE

.. autodata:: pryngles.consts.SAMPLER_GEOMETRY_CIRCLE

Sample presets are the values of `N-Spangles` for which there are already stored samples

.. autodata:: pryngles.consts.SAMPLER_SPHERE_PRESETS
   :no-value:

.. autodata:: pryngles.consts.SAMPLER_CIRCLE_PRESETS
   :no-value:

Spangler Constants
^^^^^^^^^^^^^^^^^^^^

A spangle is the discrete unit with which we sample the surface of an astrophysical body.
With the intention of modeling different surfaces in exoplanetary systems, we define the following numerical attributes to define them.

.. autodata:: pryngles.consts.SPANGLE_SOLID_ROCK

.. autodata:: pryngles.consts.SPANGLE_SOLID_ICE

.. autodata:: pryngles.consts.SPANGLE_ATMOSPHERIC

.. autodata:: pryngles.consts.SPANGLE_LIQUID

.. autodata:: pryngles.consts.SPANGLE_GRANULAR

.. autodata:: pryngles.consts.SPANGLE_GASEOUS

.. autodata:: pryngles.consts.SPANGLE_STELLAR

In order to characterize the radiation-matter interaction on each discrete unit of the body surface (`Spangle`), we can assign distinct states, coordinates and physical parameters wich will allow us to perform the computation of photometric signatures. 

.. note::

   | **1.** All lenghts and areas attributes are scaled by the `scale` factor attribute.
   | **2.** To see a detailed description of the Canonical Units (:math:`u_l, u_m, u_t`) implemented in the attributes definition, see our :doc:`system` Reference.  
   | **3.** The attributes are assign by computing them in their respective class, e.g., all Spangles coordinates are computed by the surface sampling.
   | **4.** The underscore distinction `obs/int/luz` means the reference frame in wich the attribute is computed (Observer/Intersection/Light-Source)

.. autodata:: pryngles.consts.SPANGLER_COLUMNS
   :no-value:

.. csv-table::
   :header: "Key","Data Type","Description","Units/Values"
   
   "name","str","Identification of the body having the spangler.","Body name (:data:`~ pryngles.body.Body.name`)"
   "spangle_type",":data:`~ pryngles.consts`","Type of spangle. For a list of spangle types, see the constants module.","(e.g., :any:`consts.SPANGLE_SOLID_ROCK`)"
   "geometry",":data:`~ pryngles.consts`","Geometry of the spangle. (See Sampler module constants).","(e.g., :any:`consts.SAMPLER_GEOMETRY_CIRCLE`)"
   "scale","float","The length scale of the body, e.g., for a ring this is the outer radius.","Length unit (e.g., km, :math:`u_l`)"
   "n_equ","list, `array`","Direction of the equator of the body with respect to the barycenter.","[x, y, z] (unit vector)"
   "alpha_equ","float","Zero meridian of the equatorial system.","radians"
   "w","float","Rotational angular velocity.","rad/:math:`u_t`"
   "q0","float","Initial longitude (azimuthal angle) at time :math: `t_0`. Longitude at time :math:`t` is :math:`q = q_0 + w (t - t_0)`","radians"
   "center_equ","list, `array`","Center of the body with respect to the barycenter in the body-centric equatorial system.","[x, y, z] (length units)"
   "x_equ, y_equ, z_equ","float","Cartesian coordinates of the spangle in the body-centric equatorial system.","Length units"
   "r_equ","float","Radial coordinate of the spangle in the body-centric equatorial system.","Length units"
   "q_equ","float","Longitude (azimuthal angle) of the spangle in the body-centric equatorial system.","radians"
   "f_equ","float","Latitude of the spangle in the body-centric equatorial system.","radians"
   "ns_equ","list, `array`","Unitary vector normal to the spangle in the body-centric equatorial system.","[x, y, z] (unit vector)"
   "center_ecl","list, `array`","Center of the body with respect to the barycenter in the ecliptic system.","[x, y, z] (length units)"
   "x_ecl, y_ecl, z_ecl","float","Cartesian coordinates of the spangle in the ecliptic system.","Length units"
   "wx_ecl","list, `array`","y-axis on the surface of the tangent plane to the spangle in the ecliptic system: :math:`\mathbf{w_x = (w_y \times n_s)}`.","[x, y, z] (unit vector)"
   "wy_ecl","list, `array`","y-axis on the surface of the tangent plane to the spangle in the ecliptic system: :math:`\mathbf{w_y = (n_s \times e_z)}`.","[x, y, z] (unit vector)"
   "ns_ecl","list, `array`","Unitary vector normal to the spangle, calculated in the class, in the ecliptic system.","[x, y, z] (unit vector)"
   "center_int","list, `array`","Center of the body in the intersection system.","[x, y, z] (length units)"
   "x_int, y_int, z_int","float","Cartesian coordinates of the spangle in the intersection system.","Length units"
   "ns_int","list, `array`","Unitary vector normal to the spangle, calculated in the class, in the intersection system.","[x, y, z] (unit vector)"
   "rho_int","float","Pseudo cylindrical radial coordinate of the spangle in the intersection system.","Length units"
   "az_int","float","Azimuthal angle of the spangle in the intersection system.","radians"
   "cosf_int","float",":math:`cos(\theta)` coordinate of the spangle in the intersection system.","dimensionless"
   "cos_int","float","Angle between the normal to the spangle and the direction of intersection :math:`cos(\gamma)`.","dimensionless"
   "azim_int","float","Azimuth of the direction of intersection.","radians"
   "n_int","list, `array`","Vector from the intersection origin to each spangle.","[x, y, z] (length units)"
   "n_int_ecl","list, `array`","Vector from the intersection origin to each spangle in the ecliptic system.","[x, y, z] (length units)"
   "d_int","float","Distance of the Spangle to the intersection.","Length units"
   "asp_int","float","Effective area of the spangle with respect to the intersection perspective.","Area units"
   "z_cen_int","float","z-coordinate of the center of the body to which the spangle belongs in the intersection system.","Length units"
   "hidden_mask_int","int","Bitmask of the bodies hiding the spangle (in the intersection system context). Bit ``i`` corresponds to the ``i``-th body of the spangler; see ``Spangler.body_bit``.","dimensionless (bitmask)"
   "transit_mask_int","int","Bitmask of the bodies the spangle is transiting over (in the intersection system context).","dimensionless (bitmask)"
   "transit_rho_int","float","Distance from the spangle to the center of the last body it transits over, in units of that body's scale, rounded to 4 significant figures (``NaN`` if not transiting) (in the intersection system context).","dimensionless"
   "center_obs","list, `array`","Center of the body in the observer system.","[x, y, z] (length units)"
   "x_obs, y_obs, z_obs","float","Cartesian coordinates of the spangle in the observer system.","Length units"
   "ns_obs","list, `array`","Unitary vector normal to the spangle, calculated in the class, in the observer system.","[x, y, z] (unit vector)"
   "rho_obs","float","Cylindrical radial coordinate of the spangle in the observer system.","Length units"
   "az_obs","float","Azimuthal angle of the spangle in the observer system.","radians"
   "cosf_obs","float",":math:`cos(\theta)` coordinate of the spangle in the observer system.","dimensionless"
   "cos_obs","float","Angle between the normal to the spangle and the direction of the observer :math:`cos(\epsilon)`.","dimensionless"
   "azim_obs","float","Azimuth of the direction of the observer.","radians"
   "n_obs","list, `array`","Vector from the observer origin to each spangle.","[x, y, z] (length units)"
   "d_obs","float","Distance of the Spangle to the observer.","Length units"
   "asp_obs","float","Effective area of the spangle with respect to the observer perspective.","Area units"
   "z_cen_obs","float","z-coordinate of the center of the body to which the spangle belongs in the observer system.","Length units"
   "hidden_mask_obs","int","Bitmask of the bodies hiding the spangle (in the observer system context). Bit ``i`` corresponds to the ``i``-th body of the spangler; see ``Spangler.body_bit``.","dimensionless (bitmask)"
   "transit_mask_obs","int","Bitmask of the bodies the spangle is transiting over (in the observer system context).","dimensionless (bitmask)"
   "transit_rho_obs","float","Distance from the spangle to the center of the last body it transits over, in units of that body's scale, rounded to 4 significant figures (``NaN`` if not transiting) (in the observer system context).","dimensionless"
   "beta_loc","float","Beta angle that rotates the local scattering plane to the planetary scattering plane.","radians"
   "center_luz","list, `array`","Center of the body in the light-source system.","[x, y, z] (length units)"
   "x_luz, y_luz, z_luz","float","Cartesian coordinates of the spangle in the light-source system (calculated in the class).","Length units"
   "ns_luz","list, `array`","Unitary vector normal to the spangle, calculated in the class, in the light-source system.","[x, y, z] (unit vector)"
   "rho_luz","float","Cylindrical radial coordinate of the spangle in the light-source system.","Length units"
   "az_luz","float","Azimuthal angle of the spangle in the light-source system.","radians"
   "cosf_luz","float",":math:`\mathbf{cos(\theta)}` coordinate of the spangle in the light-source system.","dimensionless"
   "cos_luz","float","Angle between the normal to the spangle and the direction of the light-source (:math:`\mathbf{cos(i)}`).","dimensionless"
   "azim_luz","float","Azimuth of the direction of the light-source.","radians"
   "n_luz","list, `array`","Vector from the light-source origin to each spangle.","[x, y, z] (length units)"
   "d_luz","float","Distance of the Spangle to the light-source.","Length units"
   "asp_luz","float","Effective area of the spangle with respect to the light-source perspective.","Area units"
   "z_cen_luz","float","z-coordinate of the center of the body to which the spangle belongs in the light-source system.","Length units"
   "hidden_mask_luz","int","Bitmask of the bodies hiding the spangle (in the light-source system context). Bit ``i`` corresponds to the ``i``-th body of the spangler; see ``Spangler.body_bit``.","dimensionless (bitmask)"
   "transit_mask_luz","int","Bitmask of the bodies the spangle is transiting over (in the light-source system context).","dimensionless (bitmask)"
   "transit_rho_luz","float","Distance from the spangle to the center of the last body it transits over, in units of that body's scale, rounded to 4 significant figures (``NaN`` if not transiting) (in the light-source system context).","dimensionless"
   "azim_obs_luz","float","Difference between the azimuth of the observer over the spangle and that of the light-source.","radians"
   "asp","float","Effective area of the spangle in 3D.","Area units"
   "dsp","float","Effective diameter of spangle, :math:`d_{sp} = 2 \sqrt{(a_{sp} / \pi)}`.","Length units"
   "scatterer","str","Hash (identifier) of the scatterer used for this spangle.","Hash string (:any:`scatterer.SCATTERER`)"
   "albedo_gray_normal","float","Wavelength-independent normal albedo.","dimensionless (0 to 1)"
   "albedo_gray_spherical","float","Wavelength-independent spherical albedo.","dimensionless (0 to 1)"
   "tau_gray_optical","float","Wavelength-independent optical depth.","dimensionless (non-negative)"
   "F, Q, U, V","float","Stokes vector components.","Flux units"
   "P","float","Degree of linear polarization, :math:`\mathbf{P = \sqrt{Q^2 + U^2} / F}`.","dimensionless"
   "emmitter","str","Hash (identifier) of the emitter used for this spangle.","Hash string"
   "Teq","float","Equilibrium temperature.","K (Kelvin)"
   "Tem","float","Emission temperature.","K (Kelvin)"
   "emmisivity","float","Emissivity (1 for a perfect black body).","dimensionless (0 to 1)"
   "unset","bool","Boolean indicating if the state has not been set.","True/False"
   "hidden","bool","Boolean indicating if the spangle is not taken into account for photometry.","True/False"
   "source","bool","Boolean indicating if the spangle belongs to a light-source (it does not reflect light).","True/False"
   "visible","bool","The spangle is visible from observer","True/False"
   "intersect","bool","Intermediate state to calculate intersections","True/False"
   "shadow","bool","The spangle is in the shadow of other spangler","True/False"
   "indirect","bool","The spangle is indirectly illuminated","True/False"
   "emit","bool","The spangle is emmitting","True/False"
   "above","bool","Intermediate state to calculate above or below state respect to ring","True/False"
   "illuminated","bool","The spangle is illuminated by the light-source","True/False"
   "transmit","bool","The spangle is illuminated but transmitting light","True/False"
   "transit","bool","The spangle is transiting","True/False"
   "occult","bool","The spangle is occulted by a light source","True/False"
   "stellar_flux","float","Incident Stellar Flux at the location of the spangle","Normalized Flux units"
   "reflected_flux","float","Reflected Flux from the spangle towards the observer","Normalized Flux units"
   "transit_flux","float","Flux blocked from the star during transit","Normalized Flux units"
   "thermal_flux","float","Thermal Emission Flux from the spangle towards the observer","Normalized Flux units"

Body Constants
^^^^^^^^^^^^^^^^^^^^

These are the numerical attributes and its defaults physical parameters to define and initialize the astrophysical body types we consider in ``pryngles`` to Spangle and create an exoplanetary system with orbital dynamic motion. `Moons` are considered as `Planet-type` body 

.. autodata:: pryngles.consts.BODY_KINDS

.. autodata:: pryngles.consts.BODY_DEFAULTS
   :no-value:

.. autodata:: pryngles.consts.STAR_DEFAULTS
   :no-value:

.. autodata:: pryngles.consts.PLANET_DEFAULTS
   :no-value:

.. autodata:: pryngles.consts.RING_DEFAULTS
   :no-value:

.. autodata:: pryngles.consts.OBSERVER_DEFAULTS
   :no-value:

.. autodata:: pryngles.consts.LEGACY_PHYSICAL_PROPERTIES
   :no-value:
