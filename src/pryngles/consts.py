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
from rebound import units
import re

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Stand alone code of the module
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import os
#Root directory
try:
    FILE=__file__
    ROOTDIR=os.path.abspath(os.path.dirname(FILE))
except:
    import IPython
    FILE=""
    ROOTDIR=os.path.abspath('')
    
IN_JUPYTER='ipykernel' in sys.modules

class Consts(object):
    """
    The ``pr.Consts`` class is where we define all physical and astronomical constants. 
    It contains the ``rebound`` imported constants and so the defined ones

    Examples
    ---------------
    >>> pr.Consts.au # AU (Astronomical Unit)
    149597870700.0
    >>> pr.Consts.msun # Solar Mass`
    1.9884754159665356e+30
    """

    # Mathematical constants
    rad=180/np.pi
    deg=1/rad
    ppm=1e6 #parts per million factor
    ppb=1e9 #parts per billion factor

    #Size of reference objects
    rearth=6378.137e3 #m, volumetric mean radius, source: 
    rsun=695700e3 #m, nominal solar radius, source: 
    rjupiter=71492e3 #m, equatorial radius, source: 
    rsaturn=60268e3 #m, equatorial radius, source:

    def get_physical():
        """
        To get all **physical** constants in ``pryngles``. 

        Returns
        -------
        :
            all_physical : list
             List containing all the strings for physical constants in package

        Examples
        --------
        >>> pr.Consts.get_physical()
        ['au', 'aus', 'cm', 'd', 'day', 'days', 'deg', 'g', 'gram', 'gyr', 'hr', 'jyr', 'kg',
        'km', 'kyr', 'm', 'massist', 'mearth', 'mjupiter', 'mmars', 'mmercury', 'mneptune', 'mpluto',
        'msaturn', 'msolar', 'msun', 'muranus', 'mvenus', 'myr', 'parsec', 'pc', 'ppb', 'ppm', 'rad', 
        'rearth', 'rjupiter', 'rsaturn', 'rsun', 's', 'solarmass', 'sunmass', 'year', 'years', 'yr', 'yrs']
        """
        import pryngles as pr
        all_constants=[]
        for key in Consts.__dict__.keys():
            patterns = "^[a-z]+$"
            if re.search(patterns,key):
                all_constants+=[key]
        return sorted(all_constants)

    def get_all():
        """
        To get all **numerical** constants in ``pryngles``.

        Returns
        -------
        :
            all_numerical : list
             List containing all the strings for numerical constants in package

        Examples
        --------
        Because some of numerical constants are grouped by modules, you can get those specifying it's module in uppercase.

        >>> [const for const in pr.Consts.get_all() if 'SPANGLER' in const]
        ['SPANGLER_AREAS', 'SPANGLER_COLUMNS', 'SPANGLER_COLUMNS_DOC', 'SPANGLER_COL_COPY', 'SPANGLER_COL_INT', 'SPANGLER_COL_LUZ',
        'SPANGLER_COL_OBS', 'SPANGLER_DEBUG_FIELDS', 'SPANGLER_EPS_BORDER', 'SPANGLER_EQUIV_COL', 'SPANGLER_KEY_ORDERING',
        'SPANGLER_KEY_SUMMARY', 'SPANGLER_LENGTHS', 'SPANGLER_SOURCE_STATES', 'SPANGLER_VECTORS', 'SPANGLER_VISIBILITY_STATES']
        """
        import pryngles as pr
        all_constants=[]
        for key in pr.__dict__.keys():
            patterns = "^[A-Z_]+$"
            if re.search(patterns,key):
                all_constants+=[key]
        return sorted(all_constants)

#Physical constants
GSI=units.convert_G(["m","s","kg"]) # G constant in SI units
for const in "times","lengths","masses":
    values=eval(f"units.{const}_SI.copy()")
    for key in values:
        exec(f"Consts.{key}=values[key]")

#For compatibility purposes with legacy
RAD=Consts.rad
DEG=Consts.deg
#Remove when legacy is retired

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module extensions
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import ctypes
DOUBLE = ctypes.c_double
PDOUBLE = ctypes.POINTER(DOUBLE)
PPDOUBLE = ctypes.POINTER(PDOUBLE)
PPPDOUBLE = ctypes.POINTER(PPDOUBLE)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module science
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SCIENCE_LIMB_NORMALIZATIONS=dict()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module orbit
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
REBOUND_ORBITAL_PROPERTIES=dict(
    #Mass
    m=0,
    #Cartesian coordinates
    x=0,y=0,z=0,vx=0,vy=0,vz=0,
    #Semi major axis, true anomaly, eccentricity
    a=1,f=0,e=0,
    #Periapsis argument, inclination, longitude of the ascending node
    omega=0,inc=0,Omega=0,
    #Mean anomaly, eccentric anomaly, time of periapsis passage
    M=0,E=0,T=0,
    #true longitude (Omega + omega + f), mean anomaly (Omega + omega + M)
    theta=0,l=0,
)
"""
`dict` : Defines the possible keys used to initialize a particle with orbital properties in ``rebound``.

- **m** (float) — Mass of the particle.
- **x**, **y**, **z** (float) — Cartesian position components.
- **vx**, **vy**, **vz** (float) — Cartesian velocity components.
- **a**, **e**, **f**, **omega**, **inc**, **Omega** (float) — Orbital Parameters
- **M**, **E** (float) — Mean and Eccentric anomaly.
- **T** (float) — Time of periapsis passage.
- **theta**, **l** (float) — True and Mean longitude.
"""


REBOUND_CARTESIAN_PROPERTIES=dict(
    x=0,y=0,z=0,vx=0,vy=0,vz=0, 
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module sampler
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# """ Sampler presets are the values of N 
#     for which there are already stored samples
# """
SAMPLER_PRESETS = ["sphere", "circle", "ring"]

SAMPLER_SPHERE_PRESETS = np.array(
    list(np.arange(100, 1000, 100))+\
    list(np.arange(1000, 2000, 200))+\
    list(np.arange(2000, 3000, 300))+\
    list(np.arange(3000, 4000, 400))+\
    list(np.arange(4000, 5000, 500))+\
    [5000]
)
"""
Examples
-------------

>>> pr.SAMPLER_SPHERE_PRESETS
array([ 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000, 1200, 1400,
        1600, 1800, 2000, 2300, 2600, 2900, 3000, 3400, 3800, 4000, 4500, 5000])
"""

SAMPLER_CIRCLE_PRESETS = np.arange(100, 6000, 100)
"""
Examples
--------------

>>>  pr.SAMPLER_CIRCLE_PRESETS
array([ 100,  200,  300,  400,  500,  600,  700,  800,  900, 1000, 1100,
        1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200,
        2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300,
        3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400,
        4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500,
        5600, 5700, 5800, 5900])
"""

SAMPLER_MIN_RING = 10

#Geometries
SAMPLER_GEOMETRY_CIRCLE=0 #:
SAMPLER_GEOMETRY_SPHERE=1 #:

SAMPLE_SHAPES=[] #:

SAMPLE_SHAPES+=["circle"]

SAMPLE_SHAPES+=["ring"]

SAMPLE_SHAPES+=["sphere"]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module spangler
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# """
#     Colors: Given in hue (0-360), level (0: black-1: white), saturation (0-1)
   
#     For colors: 
#         https://htmlcolorcodes.com/es/
   
#     When searching for colors use:
#         Plot.rgb_sample(59)
# """
#Type of spangles
SPANGLE_COLORS=dict()
__s=-1

#Spangles for planets with a rocky surface
__s+=1
SPANGLE_SOLID_ROCK=__s #:
SPANGLE_COLORS[__s]=[27,0.5,1.0]

#Spangles for planets with a rocky surface
__s+=1
SPANGLE_SOLID_ICE=__s #:
SPANGLE_COLORS[__s]=[27,0.5,1.0]

#Spangles for planets with atmospheres
__s+=1
SPANGLE_ATMOSPHERIC=__s #:
SPANGLE_COLORS[__s]=[27,0.5,1.0]

#Spangles for planets with liquid surface
__s+=1
SPANGLE_LIQUID=__s #:
SPANGLE_COLORS[__s]=[195,0.7,0.5]

#Ring or disks spangles
__s+=1
SPANGLE_GRANULAR=__s #:
SPANGLE_COLORS[__s]=[0,0.7,0.4]

#Gasseous spangle
__s+=1
SPANGLE_GASEOUS=__s #:
SPANGLE_COLORS[__s]=[27,0.5,1.0]

#Stellar spangle
__s+=1
SPANGLE_STELLAR=__s #:
SPANGLE_COLORS[__s]=[59,0.7,1.0]

#List of semitransparent spangles
SPANGLES_SEMITRANSPARENT=[SPANGLE_GRANULAR,SPANGLE_GASEOUS]

#Color of shadow
SHADOW_COLOR_LUZ=[90,0.2,1.0]
SHADOW_COLOR_OBS=[180,0.2,0.0]
SPANGLES_DARKNESS_COLOR=[225,0.3,1]

#Columns of spangling
SPANGLER_COLUMNS=odict({
    "name":"", #Identification of the body having the spangler

    #Type of spangle
    "spangle_type":SPANGLE_SOLID_ROCK, #For a list of spangle types see the constants module.
    "geometry":SAMPLER_GEOMETRY_CIRCLE, #Geometry of the spangle (see Sampler module constants)

    #Lengh-scale
    "scale":1, #The length scale of the body, eg. for a ring this is the outer radius

    #Body parameters
    "n_equ":[0,0,1], #Direction of the equator of the body with respect
    "alpha_equ":0, #Zero meridian of equatorial system
    "w":0, #Rotational angular velocity [rad/ut]
    "q0":0, #Initial time [rad], Longitude (azimutal angle) are calculated as: q = q0 + w (t - t0)

    #Coordinates of the spangle (cartesian and spherical) in the body-centric system
    "center_equ":[0,0,0],#Center of the body with respect to barycenter
    "x_equ":1,"y_equ":0,"z_equ":0, #Cartesian coordinates
    "r_equ":1,"q_equ":0,"f_equ":0, #Spherical coordinates: q: longitude, f: latitude
    "ns_equ":[0,0,1], #Unitary vector normal to the spangle

    #Coordinates of the spangle (cartesian and spherical) in the ecliptic system
    "center_ecl":[0,0,0],#Center of the body with respect to barycenter
    "x_ecl":1,"y_ecl":0,"z_ecl":0, #Cartesian coordinates of the spangle
    "wx_ecl":[1,0,0],#y-axis on the surface of the tangent plane to the spangle: wx = (wy x ns)
    "wy_ecl":[0,1,0],#y-axis on the surface of the tangent plane to the spangle: wy = (ns x ez)
    "ns_ecl":[0,0,1],#Unitary vector normal to the spangle, calculated in the class

    #Coordinates of the spangle (cartesian and spherical) in the intersection system
    "center_int":[0,0,0],#Center of the body 
    "x_int":1,"y_int":0,"z_int":0,#Cartesian coordinates
    "ns_int":[0,0,1],#Unitary vector normal to the spangle, calculated in the class
    "rho_int":1,"az_int":0,"cosf_int":0, #Pseudo cylindrical coordinates of the spangle: rho, phi, cos(theta)
    "cos_int":1, #Angle between normal to spangle and direction of intersection
    "azim_int":0, #Azimuth of the direction of intersection
    "n_int":[0,0,-np.inf],#Vector from the intersection origin to each spangle
    "n_int_ecl":[0,0,-1],#Vector from the intersection origin to each spangle in the ecliptic syste,
    "d_int":-np.inf, #Distance of the Spangle to intersection
    "asp_int":1.0, #Effective area of the spangle with respect to intersection perspective 
    "z_cen_int":0.0, #z-coordinate of the center of the body to which the spangle belows
    "hidden_by_int":"", #Which body intersect the observer or light coming to a Spangle
    "transit_over_int":"", #Which body is intersected by the Spangle (is transiting over)

    "string_int":"",#Temporal string
    
    #Coordinates of the spangle (cartesian and spherical) in the observer system
    "center_obs":[0,0,0], #Center of the body
    "x_obs":1,"y_obs":0,"z_obs":0, #Cartesian coordinates of the spangle
    "ns_obs":[0,0,1],#Unitary vector normal to the spangle, calculated in the class
    "rho_obs":1,"az_obs":0,"cosf_obs":0, #Cylindrical coordinates of the spangle: rho, phi, cos(theta)
    "cos_obs":1, #Angle between normal to spangle and direction of observer
    "azim_obs":0, #Azimuth of the direction of the observer
    "n_obs":[0,0,-np.inf],#Vector from the observer origin to each spangle
    "d_obs":-np.inf, #Distance of the Spangle to light-source
    "asp_obs":1.0, #Effective area of the spangle with respect to observer perspective 
    "z_cen_obs":0.0, #z-coordinate of the center of the body to which the spangle belows
    "hidden_by_obs":"", #Which body intersect the observer or light coming to a Spangle
    "transit_over_obs":"", #Which body is intersected by the Spangle (is transiting over)
    "beta_loc":0, #Beta angle rotates the local scattering plane to the planetary scattering plane
    
    
    #Coordinates of the spangle (cartesian and spherical) in the light-source system
    "center_luz":[0,0,0],#Center of the body
    "x_luz":1,"y_luz":0,"z_luz":0,#Calculated in the class
    "ns_luz":[0,0,1],#Unitary vector normal to the spangle, calculated in the class
    "rho_luz":1,"az_luz":0,"cosf_luz":0, #Cylindrical coordinates of the spangle: rho, phi, cos(theta)
    "cos_luz":1, #Angle between normal to spangle and direction of light-source
    "azim_luz":0, #Azimuth of the direction of the light-source
    "n_luz":[0,0,-np.inf],#Vector from the light-source origin to each spangle
    "d_luz":-np.inf, #Distance of the Spangle to light-source
    "asp_luz":1, #Effective area of the spangle with respect to light-source perspective 
    "z_cen_luz":0.0, #z-coordinate of the center of the body to which the spangle belows
    "hidden_by_luz":"", #Which body intersect the observer or light coming to a Spangle
    "transit_over_luz":"", #Which body is intersected by the Spangle (is transiting over)
    
    #Azimutal angles
    "azim_obs_luz":0,#Difference between the azimuth of the observer over the spangle and that of light-source

    #Geometrical parameters
    "asp":1.0, #Effective area of the spangle in 3D 
    "dsp":1.0, #Effective diameter of spangle, dsp = 2*(asp/pi)**0.5

    #Optical parameters
    "scatterer":"",#Hash (identifier) of the scatterer used for this spangle
    "albedo_gray_normal":1.0,#Wavelength-independent normal albedo
    "albedo_gray_spherical":1.0,#Wavelength-independent spherical albedo
    "tau_gray_optical":np.inf,#Wavelength-independent optical depth
    
    #Polarization parameters
    "F":0,"Q":0,"U":0,"V":0,"P":0, #Stokes vector components
    
    #Thermal characteristics
    "emmitter":"",#Hash (identifier) of the emmitter used for this spangle
    "Teq":273.15,#K, equilibrium temperature
    "Tem":273.15,#K, emmision temperature
    "emmisivity":1,#1 perfect black body
    
    #Special states
    "unset":True, #State has not been set
    "hidden":False, #The spangle is not taken into account for photometry
    "source":False, #The spangle belongs to a light-source (it does not reflect light)
}) #:
SPANGLER_VISIBILITY_STATES=odict({
    #Spangle state
    "visible":False, #The spangle is visible from observer
    "intersect":False, #Intermediate state to calculate intersections
    "shadow":False, #The spangle is in the shadow of other spangler
    "indirect":False, #The spangle is indirectly illuminated
    "emit":False, #The spangle is emmitting
    "above":False, #Intermediate state to calculate above or below state respect to ring
})
SPANGLER_COLUMNS.update(SPANGLER_VISIBILITY_STATES)
SPANGLER_SOURCE_STATES=odict({
    "illuminated":False, #The spangle is illuminated by the light-source
    "transmit":False, #The spangle is illuminated but transmitting light
    "transit":False, #The spangle is transiting
    "occult":False, #The spangle is occulted by a light source
})
SPANGLER_COLUMNS.update(SPANGLER_SOURCE_STATES)

SPANGLER_FLUX = odict({
    'stellar_flux':0.0, # Incident Stellar Flux at the location of the spangle
    'reflected_flux':0.0, # Reflected Flux from the spangle towards the observer
    'transit_flux':0.0, # Flux blocked from the star during transit
    'thermal_flux':0.0, # Thermal Emission Flux from the spangle towards the observer
    #Polarization parameters
    "stokes_F":0,"stokes_Q":0,"stokes_U":0,"stokes_V":0,"stokes_P":0, #Stokes vector components
    # "polarized_flux": 0.0, #Polarized flux from the spangle towards the observer
    # "polarized_degree": 0.0, #Polarization degree of the spangle
})
SPANGLER_COLUMNS.update(SPANGLER_FLUX)

SPANGLER_KEY_ORDERING=[
    
    #Spangle type
    'name','spangle_type', 
    #Coordinates
    'x_ecl', 'y_ecl', 'z_ecl', 'ns_ecl',
    #Orientation
    'azim_obs', 'n_obs', 'd_obs', 'asp_obs', 'cos_obs', 'hidden_by_obs', 'transit_over_obs', 'beta_loc',
    'azim_luz', 'n_luz', 'd_luz', 'asp_luz', 'cos_luz', 'hidden_by_luz', 'transit_over_luz',
    #Geometrical bulk properties
    'asp', 
    #Physical bulk properties
    'albedo_gray_normal', 'albedo_gray_spherical', 'tau_gray_optical', 
    'F','Q','U','V','P',
    'stellar_flux', 'reflected_flux', 'transit_flux', 'thermal_flux',
    'stokes_F','stokes_Q','stokes_U','stokes_V','stokes_P',
    'Teq', 'Tem', 'emmisivity', 
    #State
    'visible', 'shadow', 'indirect', 'emit', 
    'illuminated', 'transmit', 
    #Transit
    'transit', 'occult', 

    #Separator column
    'scale', 

    #Internal attributes
    'geometry', 
    'n_equ', 'alpha_equ', 
    'w', 'q0', 
    
    'center_equ', 
    'x_equ', 'y_equ', 'z_equ', 
    'r_equ', 'q_equ', 'f_equ', 'ns_equ', 
    
    'center_ecl', 
    'wx_ecl', 'wy_ecl', 
    
    'center_int', 
    'x_int', 'y_int', 'z_int', 'ns_int', 
    'rho_int', 'az_int', 'cosf_int', 'cos_int', 
    'azim_int', 'n_int', 'n_int_ecl', 'd_int', 'asp_int', 'z_cen_int', 'hidden_by_int', 'transit_over_int', 
    'string_int',
    
    'center_obs', 
    'x_obs', 'y_obs', 'z_obs', 'ns_obs', 
    'rho_obs', 'az_obs', 'cosf_obs', 
    'z_cen_obs',
    
    'center_luz', 
    'x_luz', 'y_luz', 'z_luz', 'ns_luz', 
    'rho_luz', 'az_luz', 'cosf_luz', 
    'z_cen_luz', 
    'azim_obs_luz', 
    
    'dsp', 
    
    #Other
    'scatterer', 'emmitter',
    
    #Internal states
    'unset', 'hidden', 'source', 'intersect', 'above', 
    ]

#These are the critical columns of the Spangler data frames to do some physics
SPANGLER_KEY_SUMMARY=[
     #Spangle type
    'name','spangle_type', 
    #Coordinates
    'x_ecl', 'y_ecl', 'z_ecl', 'ns_ecl',
    #Orientation
    'azim_obs', 'n_obs', 'd_obs', 'asp_obs', 'cos_obs', 'hidden_by_obs', 'transit_by_obs',
    'azim_luz', 'n_luz', 'd_luz', 'asp_luz', 'cos_luz', 'hidden_by_luz', 'transit_by_luz',
    #Geometrical bulk properties
    'asp', 
    #Physical bulk properties
    'albedo_gray_normal', 'albedo_gray_spherical', 'tau_gray_optical', 
    'Teq', 'Tem', 'emmisivity', 
    #State
    'visible', 'shadow', 'indirect', 'emit', 
    'illuminated', 'transmit', 
    #Transit
    'transit', 'occult', 'rho_transit',
]

#States corresponging to a given point of view
SPANGLER_EQUIV_COL=dict(obs="visible",int="intersect",luz="illuminated")

#Columns to copy when calculating visibility and illumination
SPANGLER_COL_COPY=["center","x","y","z","ns","rho","az","cosf","n","cos","azim","d","z_cen","asp"]
SPANGLER_COL_LUZ=[column+"_luz" for column in SPANGLER_COL_COPY]
SPANGLER_COL_OBS=[column+"_obs" for column in SPANGLER_COL_COPY]
SPANGLER_COL_INT=[column+"_int" for column in SPANGLER_COL_COPY]

#Spangler columns wich correspond to lengths
SPANGLER_LENGTHS=[
    "x_equ","y_equ","z_equ",
    "x_ecl","y_ecl","z_ecl",
    "x_obs","y_obs","z_obs","d_obs",
    "x_luz","y_luz","z_luz","d_luz",
    "r_equ","rho_obs","rho_luz",
    "dsp"
]

#Spangler columns which correspond to areas
SPANGLER_AREAS=[
    "asp","asp_int","asp_obs","asp_luz"
]
#Spangler columns which correspond to vectores
SPANGLER_VECTORS=[
    "center_ecl",
    "center_equ",
    "center_obs",
    "center_int",
    "n_int","n_obs","n_luz",
]

#Debugging purposes
SPANGLER_DEBUG_FIELDS=["name","spangle_type","geometry",
                     "x_obs","y_obs","z_obs","n_obs","d_obs","cos_obs",
                     "x_luz","y_luz","z_luz","n_luz","d_luz","cos_luz",
                     "x_int","y_int","z_int","n_int","d_int","cos_int"]+\
                     ["unset"]+\
                     list(SPANGLER_VISIBILITY_STATES)+list(SPANGLER_SOURCE_STATES)

#Tolerance in area of the inner border
SPANGLER_EPS_BORDER=0.01

SPANGLER_COLUMNS_DOC="""
#Columns of spangling
SPANGLER_COLUMNS=odict({
    "name":"", #Identification of the body having the spangler

    #Type of spangle
    "spangle_type":SPANGLE_SOLID_ROCK, #For a list of spangle types see the constants module.
    "geometry":SAMPLER_GEOMETRY_CIRCLE, #Geometry of the spangle (see Sampler module constants)

    #Lengh-scale
    "scale":1, #The length scale of the body, eg. for a ring this is the outer radius

    #Body parameters
    "n_equ":[0,0,1], #Direction of the equator of the body with respect
    "alpha_equ":0, #Zero meridian of equatorial system
    "w":0, #Rotational angular velocity [rad/ut]
    "q0":0, #Initial time [rad], Longitude (azimutal angle) are calculated as: q = q0 + w (t - t0)

    #Coordinates of the spangle (cartesian and spherical) in the body-centric system
    "center_equ":[0,0,0],#Center of the body with respect to barycenter
    "x_equ":1,"y_equ":0,"z_equ":0, #Cartesian coordinates
    "r_equ":1,"q_equ":0,"f_equ":0, #Spherical coordinates: q: longitude, f: latitude
    "ns_equ":[0,0,1], #Unitary vector normal to the spangle

    #Coordinates of the spangle (cartesian and spherical) in the ecliptic system
    "center_ecl":[0,0,0],#Center of the body with respect to barycenter
    "x_ecl":1,"y_ecl":0,"z_ecl":0, #Cartesian coordinates of the spangle
    "wx_ecl":[1,0,0],#y-axis on the surface of the tangent plane to the spangle: wx = (wy x ns)
    "wy_ecl":[0,1,0],#y-axis on the surface of the tangent plane to the spangle: wy = (ns x ez)
    "ns_ecl":[0,0,1],#Unitary vector normal to the spangle, calculated in the class

    #Coordinates of the spangle (cartesian and spherical) in the intersection system
    "center_int":[0,0,0],#Center of the body 
    "x_int":1,"y_int":0,"z_int":0,#Cartesian coordinates
    "ns_int":[0,0,1],#Unitary vector normal to the spangle, calculated in the class
    "rho_int":1,"az_int":0,"cosf_int":0, #Pseudo cylindrical coordinates of the spangle: rho, phi, cos(theta)
    "cos_int":1, #Angle between normal to spangle and direction of intersection
    "azim_int":0, #Azimuth of the direction of intersection
    "n_int":[0,0,-np.inf],#Vector from the intersection origin to each spangle
    "n_int_ecl":[0,0,-1],#Vector from the intersection origin to each spangle in the ecliptic syste,
    "d_int":-np.inf, #Distance of the Spangle to intersection
    "asp_int":1.0, #Effective area of the spangle with respect to intersection perspective 
    "z_cen_int":0.0, #z-coordinate of the center of the body to which the spangle belows
    "hidden_by_int":"", #Which body intersect the observer or light coming to a Spangle
    "transit_over_int":"", #Which body is intersected by the Spangle (is transiting over)

    "string_int":"",#Temporal string
    
    #Coordinates of the spangle (cartesian and spherical) in the observer system
    "center_obs":[0,0,0], #Center of the body
    "x_obs":1,"y_obs":0,"z_obs":0, #Cartesian coordinates of the spangle
    "ns_obs":[0,0,1],#Unitary vector normal to the spangle, calculated in the class
    "rho_obs":1,"az_obs":0,"cosf_obs":0, #Cylindrical coordinates of the spangle: rho, phi, cos(theta)
    "cos_obs":1, #Angle between normal to spangle and direction of observer
    "azim_obs":0, #Azimuth of the direction of the observer
    "n_obs":[0,0,-np.inf],#Vector from the observer origin to each spangle
    "d_obs":-np.inf, #Distance of the Spangle to light-source
    "asp_obs":1.0, #Effective area of the spangle with respect to observer perspective 
    "z_cen_obs":0.0, #z-coordinate of the center of the body to which the spangle belows
    "hidden_by_obs":"", #Which body intersect the observer or light coming to a Spangle
    "transit_over_obs":"", #Which body is intersected by the Spangle (is transiting over)
    
    #Coordinates of the spangle (cartesian and spherical) in the light-source system
    "center_luz":[0,0,0],#Center of the body
    "x_luz":1,"y_luz":0,"z_luz":0,#Calculated in the class
    "ns_luz":[0,0,1],#Unitary vector normal to the spangle, calculated in the class
    "rho_luz":1,"az_luz":0,"cosf_luz":0, #Cylindrical coordinates of the spangle: rho, phi, cos(theta)
    "cos_luz":1, #Angle between normal to spangle and direction of light-source
    "azim_luz":0, #Azimuth of the direction of the light-source
    "n_luz":[0,0,-np.inf],#Vector from the light-source origin to each spangle
    "d_luz":-np.inf, #Distance of the Spangle to light-source
    "asp_luz":1, #Effective area of the spangle with respect to light-source perspective 
    "z_cen_luz":0.0, #z-coordinate of the center of the body to which the spangle belows
    "hidden_by_luz":"", #Which body intersect the observer or light coming to a Spangle
    "transit_over_luz":"", #Which body is intersected by the Spangle (is transiting over)
    
    #Azimutal angles
    "azim_obs_luz":0,#Difference between the azimuth of the observer over the spangle and that of light-source

    #Geometrical parameters
    "asp":1.0, #Effective area of the spangle in 3D 
    "dsp":1.0, #Effective diameter of spangle, dsp = 2*(asp/pi)**0.5

    #Optical parameters
    "scatterer":"",#Hash (identifier) of the scatterer used for this spangle
    "albedo_gray_normal":1.0,#Wavelength-independent normal albedo
    "albedo_gray_spherical":1.0,#Wavelength-independent spherical albedo
    "tau_gray_optical":0.0,#Wavelength-independent optical depth
    
    #Thermal characteristics
    "emmitter":"",#Hash (identifier) of the emmitter used for this spangle
    "Teq":273.15,#K, equilibrium temperature
    "Tem":273.15,#K, emmision temperature
    "emmisivity":1,#1 perfect black body
    
    #Special states
    "unset":True, #State has not been set
    "hidden":False, #The spangle is not taken into account for photometry
    "source":False, #The spangle belongs to a light-source (it does not reflect light)
})
SPANGLER_VISIBILITY_STATES=odict({
    #Spangle state
    "visible":False, #The spangle is visible from observer
    "intersect":False, #Intermediate state to calculate intersections
    "shadow":False, #The spangle is in the shadow of other spangler
    "indirect":False, #The spangle is indirectly illuminated
    "emit":False, #The spangle is emmitting
    "above":False, #Intermediate state to calculate above or below state respect to ring
})
SPANGLER_COLUMNS.update(SPANGLER_VISIBILITY_STATES)
SPANGLER_SOURCE_STATES=odict({
    "illuminated":False, #The spangle is illuminated by the light-source
    "transmit":False, #The spangle is illuminated but transmitting light
    "transit":False, #The spangle is transiting
    "occult":False, #The spangle is occulted by a light source
})
SPANGLER_COLUMNS.update(SPANGLER_SOURCE_STATES)
SPANGLER_FLUX = odict({
    'stellar_flux':0.0, # Incident Stellar Flux at the location of the spangle
    'reflected_flux':0.0, # Reflected Flux from the spangle towards the observer
    'transit_flux':0.0, # Flux blocked from the star during transit
    'thermal_flux':0.0, # Thermal Emission Flux from the spangle towards the observer
})
SPANGLER_COLUMNS.update(SPANGLER_FLUX)
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module body
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BODY_KINDS=[] #:

BODY_DEFAULTS=dict()
""" 
`dict` : Defines the default keys and values used to initialize an astrophysical body.

- **name** (str or None) — The name of the body. Defaults to None.
- **name_by_kind** (bool) — If True, the name will be automatically generated based on the body's kind. Defaults to False.
- **source** (object or None) — The source object associated with this body. Defaults to None.

**Orbit Parameters:**

- **m** (float) — Mass of the body. Defaults to 1.

**Physics Parameters:**

- **radius** (float) — Radius of the body. Defaults to 1.
- **prot** (float) — Rotational period of the body. Defaults to 1.
- **i** (float) — Inclination of the rotational axis in radians. Defaults to 0.
- **roll** (float) — Roll angle for orientation. Defaults to 0.
- **alpha** (float) — Zero meridian of the body in radians. Defaults to 0.
- **q0** (float) — Initial longitude (azimuthal angle) at time t0 in radians. Defaults to 0.

**Optics Parameters:**

- **nspangles** (int) — Number of spangles used to sample the body's surface. Defaults to 1000.
- **spangle_type** (``pr.consts``) — Type of spangle used. Defaults to :data:`~ consts.SPANGLE_SOLID_ROCK`.
- **shape** (str) — The overall shape of the body. Defaults to "sphere".
- **geometry_args** (dict) — Dictionary of arguments to configure the body's geometry. Defaults to an empty dictionary.
- **seed** (int) — Random seed for spangle generation. Defaults to 0.
- **preset** (bool) — If True, use a preset distribution of spangles. Defaults to True.
- **albedo_gray_spherical** (float) — Wavelength-independent spherical albedo (0 to 1). Defaults to 1.
- **albedo_gray_normal** (float) — Wavelength-independent normal albedo (0 to 1). Defaults to 1.
- **tau_gray_optical** (float) — Wavelength-independent optical depth (non-negative). Defaults to 0.

**Legacy Parameters:**

- **primary** (object or None) — The primary body this body might orbit (legacy). Defaults to None.
- **optics** (dict) — Dictionary containing optical properties (legacy). Defaults to an empty dictionary.
- **orbit** (dict) — Dictionary containing orbital properties (legacy). Defaults to an empty dictionary.
- **physics** (dict) — Dictionary containing physical properties (legacy). Defaults to an empty dictionary.
"""
BODY_DEFAULTS.update(odict(
    
    name=None,
    name_by_kind=False,
    source=None,
    
    #Orbit
    m=1,
    center_ecl = [0,0,0], #Center of the body with respect to barycenter
    center_equ = [0,0,0], #Center of the body

    #Physics
    radius=1,
    prot=1,
    i=0, #Inclination of the rotational axis
    roll=0,
    alpha=0, #Zero meridian
    q0=0,
    
    #Optics
    nspangles=1000,
    spangle_type=SPANGLE_SOLID_ROCK,
    shape="sphere",
    geometry_args=dict(),
    seed=0,
    preset=True,
    
    albedo_gray_spherical=1,
    albedo_gray_normal=1,
    tau_gray_optical=np.inf, # Fixed from 0 to inf
    
    #Legacy
    primary=None,
    optics=dict(),
    orbit=dict(),
    physics=dict(),
))

BODY_KINDS=[]

STAR_DEFAULTS=deepcopy(BODY_DEFAULTS)
"""
`dict` : Defines the default attributes for bodies of the kind :data:`~ body.Star`. Inherits and updates defaults from :data:`~ consts.BODY_DEFAULTS`. You can also define extra Orbital Parameters included in :data:`~ consts.REBOUND_ORBITAL_PROPERTIES`

- **radius** (float) — Radius of the star. Defaults to 0.1.
- **limb_coeffs** (list of floats) — List of limb darkening coefficients [2]. Its lenght defines the model to implement [1]. **Defaults = `[]`**.
- **spangle_type** (constant) — Type of spangle used for the star's surface. Defaults to  :data:`~ consts.SPANGLE_STELLAR`.
- **shape** (str) — The overall shape of the star. Defaults to "sphere".

References
---------------
[1] Models for Limb-Darkening: https://pages.jh.edu/~dsing3/David_Sing/Limb_Darkening.html
[2] Coefficients available at: https://pages.jh.edu/~dsing3/LDfiles/LDCs.CoRot.Table1.txt
"""
STAR_DEFAULTS.update(odict(

    #Orbit: update
    #Same as body
    radius=0.1,

    #Physics: update
    #Same as Body
    T_eff = 5772, #Effective temperature [k]

    #Optical properties: update
    limb_coeffs=[],
    spangle_type=SPANGLE_STELLAR,
    shape="sphere",
))
BODY_KINDS+=["Star"]

PLANET_DEFAULTS=deepcopy(BODY_DEFAULTS)
"""
`dict` : Defines the default attributes for bodies of the kind :data:`~ body.Planet`. Inherits and updates defaults from :data:`~ consts.BODY_DEFAULTS`. You can also define extra Orbital Parameters included in :data:`~ consts.REBOUND_ORBITAL_PROPERTIES`

- **a** (float) — Semi-major axis of the planet's orbit. Defaults to 1.
- **e** (float) — Eccentricity of the planet's orbit. Defaults to 0.
- **radius** (float) — Radius of the planet. Defaults to 0.1.
- **spangle_type** (constant) — Type of spangle used for the planet's surface. Defaults to :data:`~ consts.SPANGLE_ROCK`.
- **geometry** (str) — The geometry of the planet's surface representation. Defaults to "sphere".
"""
PLANET_DEFAULTS.update(odict(

    #Orbit: update
    a=1,e=0,

    #Physics: update
    #Same as Body
    radius=0.1,

    #Optical: update
    spangle_type=SPANGLE_SOLID_ROCK,
    geometry="sphere",

    #Physics
        physics = dict(extension = 'cpixx', # Extension to compute Polarization
                       interp_method = "spline", # Interpolation method for Fourier coefficients
                       fourier_file = "fou_gasplanet_optical_50.dat", # File with Fourier coefficients
                       )
))
BODY_KINDS+=["Planet"]

"""
`dict` : Defines the default parameters for various temperature models [1] for bodies of the kind :data:`~ body.Planet`. 

References
---------------
Temperature Models are taken and adapted from the SPIDERMAN code

[1] Tom Louden, Laura Kreidberg, SPIDERMAN: an open-source code to model phase curves and secondary eclipses, Monthly Notices of the Royal Astronomical Society, Volume 477, Issue 2, June 2018, Pages 2613–2627, https://doi.org/10.1093/mnras/sty558
"""
T_MODEL_DEFAULTS = {
    "Uniform Temperature": {
        "required": ["T_planet"],
        "defaults": {"T_planet": None},
    },
    "Two Temperature": {
        "required": ["T_day", "T_night"],
        "defaults": {"T_day": None, "T_night": None},
    },
    "Zhang-Showman": {
        "required": ["xi_ratio", "T_night", "Delta_T"],
        "defaults": {"xi_ratio": None, "T_night": None, "Delta_T": None},
    },
}


RING_DEFAULTS=deepcopy(BODY_DEFAULTS)
"""
`dict` : Defines the default attributes for bodies of the kind :data:`~ body.Ring`. Inherits and updates defaults from :data:`~ consts.BODY_DEFAULTS`.

- **fi** (float) — Inner radius of the ring. Defaults to 1.5.
- **fe** (float) — Outer radius of the ring. Defaults to 2.0.
- **tau_gray_optical** (float) — Wavelength-independent optical depth of the ring. Defaults to 0.4.
- **spangle_type** (constant) — Type of spangle used for the ring particles. Defaults to :data:`~ consts.SPANGLE_GRANULAR`.
- **shape** (str) — The overall shape of the body. Defaults to "ring".

Note that Rings typically do not utilize orbital properties in the same way as other body kinds
"""
RING_DEFAULTS.update(odict(

    #Orbit: update
    #Same as Body altough ring has not orbit properties

    #Physics: update
    #Same as Body
    fi=1.5,
    fe=2.0,
    taur=0.4,

    #Optics: update
    spangle_type=SPANGLE_GRANULAR,
    shape="ring",

    #Physics
    physics = dict(extension = 'cpixx', # Extension to compute Polarization
                   interp_method = "bilinear", # Interpolation method for Fourier coefficients
                   fourier_file = "fou_ring_0_4_0_8.dat", # File with Fourier coefficients
                   )
))
BODY_KINDS+=["Ring"]

OBSERVER_DEFAULTS=deepcopy(BODY_DEFAULTS)
"""
`dict` : Defines the default attributes for bodies of the kind 'Observer'. Inherits and updates defaults from :data:`~ consts.BODY_DEFAULTS`.

- **lamb** (float) — Ecliptic longitude of the observer in radians. Defaults to 0.
- **beta** (float) — Ecliptic latitude of the observer in radians. Defaults to 0.
"""
OBSERVER_DEFAULTS.update(odict(
    lamb=0,
    beta=0,
))
BODY_KINDS+=["Observer"]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module system
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LEGACY_PHYSICAL_PROPERTIES=dict(
    #Albedos
    AS=1,AL=1,
    #Ring geometrical opacity
    taug=1.0, #Geometrical opacity
    diffeff=1.0, #Diffraction efficiency
    #Law of diffuse reflection on ring surface
    reflection_rings_law=lambda x,y:x,
    #Observations wavelength
    wavelength=550e-9,
    #Ring particle propeties (see French & Nicholson, 2000)
    particles=dict(q=3,s0=100e-6,smin=1e-2,smax=1e2,Qsc=1,Qext=2),
    #Stellar limb darkening
    limb_cs=[],
)
"""
`dict` : Defines legacy physical properties used from the old-version interface :data:`legacy.RingedPlanet`.

- **AS** (float) — Single scattering albedo. Defaults to 1.
- **AL** (float) — Lambertian albedo. Defaults to 1.
- **taug** (float) — Geometrical opacity of rings. Defaults to 1.0.
- **diffeff** (float) — Diffraction efficiency of rings. Defaults to 1.0.
- **reflection_rings_law** (callable) — Function defining the law of diffuse reflection on the ring surface. Takes two arguments (likely angles) and returns a reflection coefficient. Defaults to a lambda function returning the first argument.
- **wavelength** (float) — Observation wavelength in meters. Defaults to 550e-9.
- **limb_cs** (list - array) — Stellar limb darkening coefficients (legacy). Defaults to an empty list.
- **particles** (dict) — Dictionary containing properties of ring particles
    - **q** (int) — Power-law exponent for the size distribution. Defaults to 3.
    - **s0** (float) — Reference particle size in meters. Defaults to 100e-6.
    - **smin** (float) — Minimum particle size in meters. Defaults to 1e-2.
    - **smax** (float) — Maximum particle size in meters. Defaults to 1e2.
    - **Qsc** (float) — Scattering efficiency. Defaults to 1.
    - **Qext** (float) — Extinction efficiency. Defaults to 2.
"""


DETECTOR_PROPERTIES = dict(
    # Waveband
    wavelength_min = 500e-9, # [m]
    wavelength_max = 700e-9, # [m] 
    # Apperture
    apperture = 0.5, # [m]
    # Quantum Efficiency
    quantum_eff = 1,
    # Time of Cadence    
    t_cadence = 10*60, # Minutes [s]
    # Observer Distance
    distance = 1*1e3*Consts.pc # Kilo Parsec [m]
)
"""
`dict` : Defines the properties for the detector used in :data:`~ system.System`.

- **wavelength_min** (float) — Minimum wavelength of the detector's sensitivity range in meters [m]. Defaults to 500 nm.
- **wavelength_max** (float) — Maximum wavelength of the detector's sensitivity range in meters [m]. Defaults to 700 nm.
- **apperture** (float) — Aperture size of the detector in meters [m]. Defaults to 0.5 m.
- **quantum_eff** (float) — Quantum efficiency of the detector. Defines the ratio of detected photons to incident photons. Defaults to 1.
- **t_cadence** (float) — Time of cadence for observations in seconds [s]. Defaults to 10 minutes.
- **distance** (float) — Distance from the observer to the system being observed in meters [m]. Defaults to 1 kiloparsec.
"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Constants of module scatterer
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
try:
    SCATTERERS_CATALOGUE
except:
    SCATTERERS_CATALOGUE=dict()
