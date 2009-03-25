"""UFL - Unified Form Language
---------------------------

NB! This is a preliminary prototype version of UFL, which is still work in progress.

This module contains a model implementation of the Unified Form Language.

- The user interface is in the global UFL namespace:
    from ufl import *

- The underlying classes an UFL expression tree
  is built from can be imported by:
    from ufl.classes import *
  These are considered implementation details 
  and should not be used in form definitions.

- Various algorithms for working with UFL
  expression trees can be found in:
    from ufl.algorithms import *
  These are considered implementation details 
  and should not be used in form definitions.


A full manual should later become available at:

http://www.fenics.org/ufl/

But at the moment we only have some unfinished wiki pages with preliminary and incomplete feature descriptions.
"""

__version__ = "0.1"
__authors__ = "Martin Sandve Alnes and Anders Logg"
__copyright__ = "Copyright (C) 2008-2009 " + __authors__
__license__  = "GNU GPL version 3 or any later version"
__date__ = "2008-03-14 -- 2009-03-25"

########## README
# Imports here should be what the user sees when doing "from ufl import *",
# which means we should _not_ import f.ex. "Grad", but "grad".
# This way we expose the language, the operation "grad", but less
# of the implementation, the particular class "Grad".
##########

# Utility functions (product is the counterpart of the built-in
# python function sum, can be useful for users as well?)
from ufl.common import product

# Output control
from ufl.log import get_handler, get_logger, set_handler, set_level, \
    UFLException, DEBUG, INFO, WARNING, ERROR, CRITICAL

# Finite elements classes
from ufl.finiteelement import FiniteElementBase, FiniteElement, \
    MixedElement, VectorElement, TensorElement, ElementUnion

# Hook to extend predefined element families
from ufl.elements import register_element #, ufl_elements

# Basis functions
from ufl.basisfunction import BasisFunction, TestFunction, TrialFunction, \
    BasisFunctions, TestFunctions, TrialFunctions

# Coefficient functions
from ufl.function import Function, Functions, \
    Constant, VectorConstant, TensorConstant

# Literal constants
from ufl.constantvalue import Identity

# Indexing of tensor expressions
from ufl.indexing import Index, indices

# Special functions for expression base classes
# (ensure this is imported, since it attaches operators to Expr)
import ufl.exproperators as __exproperators

# Containers for expressions with value rank > 0
from ufl.tensors import as_tensor, as_vector, as_matrix, relabel
from ufl.tensors import unit_vector, unit_vectors, unit_matrix, unit_matrices

# Types for geometric quantities
# Currently disabled to encourage use of syntax
# x = cell.x; n = cell.n
from ufl.geometry import Cell # , SpatialCoordinate, FacetNormal 

# Operators
from ufl.operators import transpose, outer, inner, dot, cross, det, \
                       inv, tr, dev, cofac, skew, \
                       sqrt, exp, ln, cos, sin, \
                       eq, ne, le, ge, lt, gt, conditional, sign, \
                       jump, avg, \
                       variable, \
                       Dx, diff, grad, div, curl, rot #, Dt

# Form class
from ufl.form import Form

# Integral classes
from ufl.integral import Integral

# Representations of transformed forms
from ufl.formoperators import derivative, action, rhs, lhs, functional, adjoint #, dirichlet_functional

# Predefined convenience objects
from ufl.objects import interval, triangle, tetrahedron, quadrilateral, hexahedron, \
                     i, j, k, l, p, q, r, s, \
                     dx, ds, dS

# Constants
from math import e, pi

# Split function
from ufl.split_functions import split

