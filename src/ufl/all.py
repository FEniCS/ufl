#!/usr/bin/env python

"""
Convenience file to import all parts of the language.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 8th 2008"

# base system (base class and all subclasses involved in operators on the base class)
from base import *

# representations of transformed forms
from forms import *

# finite elements
from elements import *

# basisfunctions and coefficients
from arguments import *

# differential operators (div, grad, curl etc)
from differential import *

# compound tensoralgebra operations (dot, trace, etc)
from tensoralgebra import *

# mathematical functions (sin, cos, exp, ln etc.)
from function import *

# types for geometric quantities
from geometry import *

# predefined convenience objects like I, n, h and i,j,k,l
from objects import *

