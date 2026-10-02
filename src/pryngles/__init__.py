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

import warnings

from pryngles.body import Body, Detector, Observer, Planet, Ring, Star
from pryngles.common import PrynglesCommon, Verbose
from pryngles.consts import Consts
from pryngles.extensions import ExtensionUtil, FourierCoefficients, StokesScatterer
from pryngles.legacy import CanonicalUnits, Conf, Const, Extra, RingedPlanet, Sample, Util
from pryngles.orbit import Orbit, OrbitUtil, Orbody
from pryngles.plot import Plot
from pryngles.sampler import Sampler
from pryngles.scatterer import (
    BlackBodySurface,
    LambertianGrayAtmosphere,
    LambertianGraySurface,
    NeutralSurface,
    Scatterer,
)
from pryngles.science import Plane
from pryngles.spangler import Spangler
from pryngles.system import System
from pryngles.version import version

__all__ = [
    "Body",
    "Star",
    "Planet",
    "Ring",
    "Observer",
    "Detector",
    "Verbose",
    "PrynglesCommon",
    "Consts",
    "ExtensionUtil",
    "FourierCoefficients",
    "StokesScatterer",
    "Const",
    "CanonicalUnits",
    "Util",
    "Conf",
    "Sample",
    "RingedPlanet",
    "Extra",
    "Orbody",
    "Orbit",
    "OrbitUtil",
    "Plot",
    "Sampler",
    "Scatterer",
    "NeutralSurface",
    "BlackBodySurface",
    "LambertianGraySurface",
    "LambertianGrayAtmosphere",
    "Plane",
    "Spangler",
    "System",
    "version",
]


warnings.filterwarnings("ignore")

"""Show a welcome message when importing the package."""
print(f"Welcome to pryngles v{version}!")
