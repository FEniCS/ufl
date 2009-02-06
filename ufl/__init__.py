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
__date__ = "2008-03-14 -- 2009-02-04"

########## README
# Imports here should be what the user sees when doing "from ufl import *",
# which means we should _not_ import f.ex. "Grad", but "grad".
# This way we expose the language, the operation "grad", but less
# of the implementation, the particular class "Grad".
##########

# utility functions (product is the counterpart of the built-in python function sum, can be useful for users as well)
from ufl.common import product

# output control
from ufl.log import get_handler, get_logger, set_handler, UFLException

# finite elements classes
from ufl.finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement

# hook to extend predefined element families
from ufl.elements import register_element #, ufl_elements

# basis functions
from ufl.basisfunction import BasisFunction, TestFunction, TrialFunction, \
                           BasisFunctions, TestFunctions, TrialFunctions

# coefficient functions
from ufl.function import Function, Functions, \
                      Constant, VectorConstant, TensorConstant

# literal constants
from ufl.constantvalue import Identity

# indexing of tensor expressions
from ufl.indexing import Index, indices

# special functions for expression base classes (ensure this is imported, since it attaches operators to Expr)
from ufl import exproperators as __exproperators

# containers for expressions with value rank > 0
from ufl.tensors import as_tensor, as_vector, as_matrix, relabel
from ufl.tensors import unit_vector, unit_vectors, unit_matrix, unit_matrices

# types for geometric quantities
from ufl.geometry import Cell # , SpatialCoordinate, FacetNormal # use cell = Cell("triangle"); x = cell.x(); n = cell.n()

# operators
from ufl.operators import transpose, outer, inner, dot, cross, det, inv, tr, dev, cofac, skew, \
                       sqrt, exp, ln, cos, sin, \
                       eq, ne, le, ge, lt, gt, conditional, sign, \
                       jump, avg, \
                       variable, \
                       Dx, diff, grad, div, curl, rot #, Dt

# form class
from ufl.form import Form

# integral classes
from ufl.integral import Integral

# representations of transformed forms
from ufl.formoperators import derivative, action, rhs, lhs, functional, adjoint #, dirichlet_functional

# predefined convenience objects
from ufl.objects import interval, triangle, tetrahedron, quadrilateral, hexahedron, \
                     i, j, k, l, p, q, r, s, \
                     dx, ds, dS

# constants
from math import e, pi
