"""
UFL - Unified Form Language
---------------------------

NB! This is a preliminary prototype version of UFL, which is still work in progress.

This module contains a model implementation of the Unified Form Language,
consisting of two main parts:

- The user interface:
from ufl import *

- Various algorithms for converting, inspecting and transforming UFL expression trees:
from ufl.algorithms import *


A full manual should later become available at:

http://www.fenics.org/ufl/

But at the moment we only have some unfinished wiki pages with preliminary and incomplete feature descriptions.

"""

from __future__ import absolute_import

__version__ = "0.1"
__authors__ = "Martin Sandve Alnes and Anders Logg"
__copyright__ = __authors__ + " (2008)"
__licence__ = "LGPL" # TODO: which licence?
__date__ = "2008-03-14 -- 2008-08-18"

# DEBUGGING: Enable PyChecker to find potential bugs by static code analysis:
#import os, sys, glob, numpy, logging # PyChecker skips previously loaded modules
#import pychecker.checker


########## README
# Imports here should be what the user sees when doing "from ufl import *",
# which means we should _not_ import f.ex. "Grad", but "grad".
# This way we expose the language, the operation "grad", but less
# of the implementation, the particular class "Grad".
##########


# utility functions (product is the counterpart of the built-in python function sum, can be useful for users as well)
from .common import product

# permuation tools
#from .permutation import compute_indices, compute_permutations, compute_permutation_pairs, compute_sign, compute_order_tuples

# output control
#from .output import get_handler, get_logger, set_handler
from .output import UFLException, ufl_debug, ufl_info, ufl_warning, ufl_error, ufl_assert

# base system (expression base classes)
from .base import UFLObject, Terminal, Number, Compound
#from .base import is_python_scalar, is_scalar, is_true_scalar

# variable class
#from .variable import Variable

# finite elements classes
#from .finiteelement import FiniteElementBase
from .finiteelement import FiniteElement, MixedElement, VectorElement, TensorElement

# predefined element families
from .elements import register_element #, ufl_elements

# basisfunctions and coefficients
from .basisfunctions import BasisFunction, TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions, Function, Constant

# types for geometric quantities
from .geometry import FacetNormal

# indexing of tensor expressions
#from .indexing import Index, FixedIndex, AxisType, MultiIndex, Indexed
#from .indexing import as_index, as_index_tuple, extract_indices

# basic algebra operators
#from .algebra import Sum, Product, Division, Power, Mod, Abs

# restriction operators
from .restriction import Restricted, PositiveRestricted, NegativeRestricted

# special functions for expression base classes
from . import baseoperators as __baseoperators

# "container" classes for expressions with value rank > 0
#from .tensors import ListVector, ListMatrix
from .tensors import Tensor, Vector, Matrix

# tensor algebra operators
#from .tensoralgebra import Transpose, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from .tensoralgebra import Identity

# mathematical functions
#from .mathfunctions import MathFunction
from .mathfunctions import sqrt, exp, ln, cos, sin

# differentiation operators
#from .differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot

# operators
from .operators import transpose, outer, inner, dot, cross, det, inv, tr, dev, cofac, Dx, Dt, diff, grad, div, curl, rot, jump, avg, variable

# form class
from .form import Form

# integral classes
from .integral import Integral

# representations of transformed forms
from .formoperators import Derivative, Action, Rhs, Lhs, rhs, lhs

# predefined convenience objects
from .objects import n, i, j, k, l, p, q, r, s
from .objects import dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9
from .objects import ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9
from .objects import dS0, dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8, dS9
from .objects import dx, ds, dS
from .objects import interval, triangle, tetrahedron, quadrilateral, hexahedron

# Constants
from math import e, pi

# algorithms
from . import algorithms

