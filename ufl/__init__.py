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


A very brief overview of the language contents follows:

Cells and Euclidean spaces:
    interval, triangle, tetrahedron, quadrilateral, hexahedron,
    R1, R2, R3,
    Cell, Space

Elements:
    FiniteElement, MixedElement, VectorElement, TensorElement
    ElementUnion, RestrictedElement

Arguments:
    BasisFunction, TestFunction, TrialFunction

Coefficients:
    Coefficient, Constant, VectorConstant, TensorConstant

Splitting form arguments in mixed spaces:
    split

Literal constants:
    Identity

Geometric quantities:
    SpatialCoordinate, FacetNormal

Indices:
    Index, indices,
    i, j, k, l, p, q, r, s

Scalar to tensor expression conversion:
    as_tensor, as_vector, as_matrix

Unit vectors and matrices:
    unit_vector, unit_vectors,
    unit_matrix, unit_matrices

Tensor algebra operators:
    outer, inner, dot, cross,
    transpose, tr,
    det, inv, cofac,
    dev, skew, sym,

Differential operators:
    variable, diff,
    Dx, grad, div, curl, rot, Dn

Nonlinear functions:
    abs, sign, sqrt, exp, ln, cos, sin, tan, acos, asin, atan

Discontinuous Galerkin operators:
    jump, avg, v('+'), v('-')

Conditional operators:
    eq, ne, le, ge, lt, gt, conditional,

Integral measures:
    dx, ds, dS, dE

Form transformations:
    rhs, lhs, system, functional,
    replace, adjoint, action, energy_norm,
    sensitivity_rhs, derivative

"""

__version__ = "0.5.2"
__authors__ = "Martin Sandve Alnes and Anders Logg"
__copyright__ = "Copyright (C) 2008-2010 " + __authors__
__license__  = "GNU GPL version 3 or any later version"
__date__ = "2008-03-14"

# Modified by Kristian Oelgaard, 2009
# Modified by Anders Logg, 2009.
# Last changed: 2010-02-01

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
from ufl.geometry import Cell, Space, SpatialCoordinate, FacetNormal

# Finite elements classes
from ufl.finiteelement import FiniteElementBase, FiniteElement, \
    MixedElement, VectorElement, TensorElement, ElementUnion, ElementRestriction

# Hook to extend predefined element families
from ufl.elementlist import register_element, show_elements #, ufl_elements

# Arguments
from ufl.argument import Argument, TestFunction, TrialFunction, \
                         Arguments, TestFunctions, TrialFunctions

# Coefficients
from ufl.coefficient import Coefficient, Coefficients, \
                            Constant, VectorConstant, TensorConstant

# Split function
from ufl.split_functions import split

# Literal constants
from ufl.constantvalue import Identity, as_ufl

# Indexing of tensor expressions
from ufl.indexing import Index, indices

# Special functions for expression base classes
# (ensure this is imported, since it attaches operators to Expr)
import ufl.exproperators as __exproperators

# Containers for expressions with value rank > 0
from ufl.tensors import as_tensor, as_vector, as_matrix, relabel
from ufl.tensors import unit_vector, unit_vectors, unit_matrix, unit_matrices

# Operators
from ufl.operators import rank, shape, \
                       outer, inner, dot, cross, \
                       det, inv, cofac, \
                       transpose, tr, dev, skew, sym, \
                       sqrt, exp, ln, cos, sin, tan, acos, asin, atan, \
                       eq, ne, le, ge, lt, gt, conditional, sign, \
                       variable, diff, \
                       Dx,  grad, div, curl, rot, Dn, \
                       jump, avg

# Lifting
from ufl.lifting import LiftingFunction, LiftingOperator

# Form class
from ufl.form import Form

# Integral classes
from ufl.integral import Integral, Measure, register_domain_type

# Representations of transformed forms
from ufl.formoperators import replace, derivative, action, energy_norm, rhs, lhs, system, functional, adjoint, sensitivity_rhs #, dirichlet_functional

# Predefined convenience objects
from ufl.objects import vertex, interval, triangle, tetrahedron, quadrilateral, hexahedron, facet,\
                     i, j, k, l, p, q, r, s, \
                     R1, R2, R3, \
                     dx, ds, dS, dE, dc

# Useful constants
from math import e, pi

