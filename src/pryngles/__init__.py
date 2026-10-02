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
from pryngles.consts import *

# Utility modules
from pryngles.common import Verbose, PrynglesCommon, VERB_NONE, VERB_SIMPLE, VERB_SYSTEM, VERB_VERIFY, VERB_DEEP, VERB_ALL
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
