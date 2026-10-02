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
from IPython import get_ipython
from IPython.display import HTML, Image, display

# --------------------------------------------------
# Stand alone code of the module
# --------------------------------------------------

from pryngles.version import *

# Constants
from pryngles.consts import *

# Utility modules
from pryngles.common import Verbose, PrynglesCommon
from pryngles.misc import *
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

# Reset verbosity
Verbose.VERBOSITY = VERB_NONE

# Alias
verbose = Verbose.print

# This aliases does not work in modules
print_df = Misc.print_df
sci = Science
