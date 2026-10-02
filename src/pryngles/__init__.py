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

import sys
import unittest
import warnings
from collections import OrderedDict as odict
from copy import deepcopy

import sigfig

warnings.filterwarnings("ignore")

# JupDev: Jupyter compatibility
import IPython.core.autocall as autocall
from IPython.display import Image

# --------------------------------------------------
# Stand alone code of the module
# --------------------------------------------------

from pryngles.version import *

# Constants
from pryngles.consts import (
    BODY_DEFAULTS,
    BODY_KINDS,
    DEG,
    DETECTOR_PROPERTIES,
    GSI,
    IN_JUPYTER,
    LEGACY_PHYSICAL_PROPERTIES,
    OBSERVER_DEFAULTS,
    PLANET_DEFAULTS,
    RAD,
    REBOUND_CARTESIAN_PROPERTIES,
    REBOUND_ORBITAL_PROPERTIES,
    RING_DEFAULTS,
    ROOTDIR,
    SAMPLER_CIRCLE_PRESETS,
    SAMPLER_GEOMETRY_CIRCLE,
    SAMPLER_GEOMETRY_SPHERE,
    SAMPLER_MIN_RING,
    SAMPLER_PRESETS,
    SAMPLER_SPHERE_PRESETS,
    SAMPLE_SHAPES,
    SCATTERERS_CATALOGUE,
    SCIENCE_LIMB_NORMALIZATIONS,
    SHADOW_COLOR_LUZ,
    SHADOW_COLOR_OBS,
    SPANGLER_AREAS,
    SPANGLER_COLUMNS,
    SPANGLER_COLUMNS_DOC,
    SPANGLER_COL_COPY,
    SPANGLER_COL_INT,
    SPANGLER_COL_LUZ,
    SPANGLER_COL_OBS,
    SPANGLER_DEBUG_FIELDS,
    SPANGLER_EPS_BORDER,
    SPANGLER_EQUIV_COL,
    SPANGLER_FLUX,
    SPANGLER_KEY_ORDERING,
    SPANGLER_KEY_SUMMARY,
    SPANGLER_LENGTHS,
    SPANGLER_SOURCE_STATES,
    SPANGLER_VECTORS,
    SPANGLER_VISIBILITY_STATES,
    SPANGLES_DARKNESS_COLOR,
    SPANGLES_SEMITRANSPARENT,
    SPANGLE_ATMOSPHERIC,
    SPANGLE_COLORS,
    SPANGLE_GASEOUS,
    SPANGLE_GRANULAR,
    SPANGLE_LIQUID,
    SPANGLE_SOLID_ICE,
    SPANGLE_SOLID_ROCK,
    SPANGLE_STELLAR,
    STAR_DEFAULTS,
    T_MODEL_DEFAULTS,
)

# Utility modules
from pryngles.common import (
    Verbose,
    PrynglesCommon,
    VERB_NONE,
    VERB_SIMPLE,
    VERB_SYSTEM,
    VERB_VERIFY,
    VERB_DEEP,
    VERB_ALL,
)
from pryngles.misc import print_df, DATA_INDEX
from pryngles.extensions import *
from pryngles.science import *
from pryngles.plot import *
from pryngles.orbit import *
from pryngles.scatterer import *

# Legacy module
from pryngles.legacy import *

# Core modules
from pryngles.sampler import *
from pryngles.spangler import *
from pryngles.body import *
from pryngles.system import *


def _welcome():
    """Show a welcome message when importing the package."""
    print(f"Welcome to pryngles v{version}!")


_welcome()

# This aliases does not work in modules
sci = Science
