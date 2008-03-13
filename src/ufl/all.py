#!/usr/bin/env python

"""
Convenience file to import all parts of the language.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 13th 2008"

# base system (expression base class and all subclasses involved in operators on the base class)
from base import *

# form class
from form import *

# integral classes
from integral import *

# representations of transformed forms
from formoperators import *

# finite elements
from elements import *

# basisfunctions and coefficients
from basisfunctions import *

# differential operators (div, grad, curl etc)
from diffoperators import *

# compound tensoralgebra operations (dot, trace, etc)
from tensoroperators import *

# mathematical functions (sin, cos, exp, ln etc.)
from mathfunctions import *

# types for geometric quantities
from geometry import *

# predefined convenience objects like I, n, h and i,j,k,l
from objects import *

