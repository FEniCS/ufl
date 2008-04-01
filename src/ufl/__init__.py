#!/usr/bin/env python
"""
UFL - Unified Form Language
---------------------------

NB! This is a preliminary prototype version of UFL, which is still work in progress.

This module contains a model implementation of the Unified Form Language,
consisting of two main parts:

- The user interface:
from ufl import *

- Various utilities for converting, inspecting and transforming UFL expression trees:
from ufl.utilities import *


A full manual should later become available at:

http://www.fenics.org/ufl/

But at the moment we only have some unfinished wiki pages with preliminary and incomplete feature descriptions.

"""

__version__ = "0.1"
__authors__ = "Martin Sandve Alnes and Anders Logg"
__copyright__ = __authors__ + " (2008)"
__licence__ = "GPL" # TODO: which licence?
__date__ = "2008-03-14 -- 2008-04-01"

# form language
from all import *

# algorithms
import utilities
