"""The Unified Form Language is an embedded domain specific
language for definition of variational forms intended for
finite element discretization. More precisely, it defines
a fixed interface for choosing finite element spaces and
defining expressions for weak forms in a notation close
to mathematical notation.

This python module contains the language as well as
algorithms to work with it.

- To import the language, type:
      from ufl import *

- To import the underlying classes an UFL expression tree
  is built from, type:
      from ufl.classes import *

- Various algorithms for working with UFL expression trees
  can be found in:
      from ufl.algorithms import *

The classes and algorithms are considered implementation 
details and should not be used in form definitions.

For more details on the language, see the manual or the
wiki pages at

  http://www.fenics.org/wiki/UFL/

The development version can be found in the repository at

  http://www.fenics.org/hg/ufl/

"""

__version__ = "0.3"
__authors__ = "Martin Sandve Alnes and Anders Logg"
__copyright__ = "Copyright (C) 2008-2009 " + __authors__
__license__  = "GNU GPL version 3 or any later version"
__date__ = "2008-03-14 -- 2009-04-20"

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
from ufl.log import get_handler, get_logger, set_handler, set_level, add_logfile, \
    UFLException, DEBUG, INFO, WARNING, ERROR, CRITICAL

# Types for geometric quantities
# Currently disabled to encourage use of syntax
# x = cell.x; n = cell.n
from ufl.geometry import Cell # , SpatialCoordinate, FacetNormal 

# Finite elements classes
from ufl.finiteelement import FiniteElementBase, FiniteElement, \
    MixedElement, VectorElement, TensorElement, ElementUnion

# Hook to extend predefined element families
from ufl.elementlist import register_element, show_elements #, ufl_elements

# Basis functions
from ufl.basisfunction import BasisFunction, TestFunction, TrialFunction, \
    BasisFunctions, TestFunctions, TrialFunctions

# Coefficient functions
from ufl.function import Function, Functions, \
    Constant, VectorConstant, TensorConstant

# Split function
from ufl.split_functions import split

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
from ufl.formoperators import derivative, action, energy_norm, rhs, lhs, system, functional, adjoint, sensitivity_rhs #, dirichlet_functional

# Predefined convenience objects
from ufl.objects import interval, triangle, tetrahedron, quadrilateral, hexahedron, \
                     i, j, k, l, p, q, r, s, \
                     dx, ds, dS

# Useful constants
from math import e, pi
