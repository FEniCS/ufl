#!/usr/bin/env python

"""
Convenience file to import all parts of the language.
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-05-15"

# base system (expression base class and all subclasses involved in operators on the base class)
from base import *

# form class
from form import *

# integral classes
from integral import *

# representations of transformed forms
from formoperators import *

# finite elements classes
from finiteelement import *

# predefined element families
from elements import ufl_elements, register_element

# basisfunctions and coefficients
from basisfunctions import *

# "container" classes for expressions with value rank > 0
from tensors import *

# operators
from operators import *

# mathematical functions (sin, cos, exp, ln etc.)
from mathfunctions import *

# types for geometric quantities
from geometry import *

# predefined convenience objects like I, n, h and i,j,k,l
from objects import *

# TODO: describe these
from differentiation import *
from indexing import *
from tensoralgebra import *

