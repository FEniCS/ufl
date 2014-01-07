"""The Unified Form Language is an embedded domain specific language
for definition of variational forms intended for finite element
discretization. More precisely, it defines a fixed interface for choosing
finite element spaces and defining expressions for weak forms in a
notation close to mathematical notation.

This Python module contains the language as well as algorithms to work
with it.

* To import the language, type::

    from ufl import *

* To import the underlying classes an UFL expression tree is built
  from, type::

    from ufl.classes import *

* Various algorithms for working with UFL expression trees can be
  found in::

    from ufl.algorithms import *

The classes and algorithms are considered implementation details and
should not be used in form definitions.

For more details on the language, see

  http://www.fenicsproject.org

and

  http://arxiv.org/abs/1211.4047

The development version can be found in the repository at

  https://www.bitbucket.org/fenics-project/ufl

A very brief overview of the language contents follows:

* Domains::

    Domain, Region

* Cells::

    Cell,
    interval, triangle, tetrahedron,
    quadrilateral, hexahedron,
    cell1D, cell2D, cell3D,

* Elements::

    FiniteElement,
    MixedElement, VectorElement, TensorElement
    EnrichedElement, RestrictedElement,
    TensorProductElement

* Arguments::

    Argument, TestFunction, TrialFunction

* Coefficients::

    Coefficient, Constant, VectorConstant, TensorConstant

* Splitting form arguments in mixed spaces::

    split

* Literal constants::

    Identity, PermutationSymbol

* Geometric quantities::

    SpatialCoordinate, FacetNormal,
    CellVolume, Circumradius, CellSurfaceArea,
    FacetArea, MinFacetEdgeLength, MaxFacetEdgeLength, FacetDiameter,
    LocalCoordinate, GeometryJacobi,
    GeometryJacobiDeterminant, InverseGeometryJacobi

* Indices::

    Index, indices,
    i, j, k, l, p, q, r, s

* Scalar to tensor expression conversion::

    as_tensor, as_vector, as_matrix

* Unit vectors and matrices::

    unit_vector, unit_vectors,
    unit_matrix, unit_matrices

* Tensor algebra operators::

    outer, inner, dot, cross, perp,
    det, inv, cofac,
    transpose, tr, diag, diag_vector,
    dev, skew, sym

* Elementwise tensor operators::

    elem_mult, elem_div, elem_pow, elem_op

* Differential operators::

    variable, diff,
    grad, div, nabla_grad, nabla_div,
    Dx, Dn, curl, rot

* Nonlinear functions::

    Max, Min,
    abs, sign, sqrt,
    exp, ln, erf,
    cos, sin, tan,
    acos, asin, atan, atan_2,
    cosh, sinh, tanh,
    bessel_J, bessel_Y, bessel_I, bessel_K

* Discontinuous Galerkin operators:
    jump, avg, v('+'), v('-'), cell_avg, facet_avg

* Conditional operators::

    eq, ne, le, ge, lt, gt,
    <, >, <=, >=,
    And, Or, Not,
    conditional

* Integral measures::

    dx, ds, dS, dP, dE, dc

* Form transformations::

    rhs, lhs, system, functional,
    replace, adjoint, action, energy_norm,
    sensitivity_rhs, derivative
"""

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Kristian B. Oelgaard, 2009, 2011
# Modified by Anders Logg, 2009.
# Modified by Johannes Ring, 2014.
#
# Last changed: 2014-01-07

__version__ = "1.3.0"

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
from ufl.geometry import (Cell, ProductCell,
     SpatialCoordinate, FacetNormal,
     CellVolume, Circumradius, CellSurfaceArea,
     FacetArea, MinFacetEdgeLength, MaxFacetEdgeLength, FacetDiameter,
     LocalCoordinate, GeometryJacobi,
     GeometryJacobiDeterminant, InverseGeometryJacobi)

# Types for domain description
from ufl.domains import Domain, Region

# Finite elements classes
from ufl.finiteelement import FiniteElementBase, FiniteElement, \
    MixedElement, VectorElement, TensorElement, EnrichedElement, \
    RestrictedElement, TensorProductElement

# Hook to extend predefined element families
from ufl.finiteelement.elementlist import register_element, show_elements #, ufl_elements

# Arguments
from ufl.argument import Argument, TestFunction, TrialFunction, \
                         Arguments, TestFunctions, TrialFunctions

# Coefficients
from ufl.coefficient import Coefficient, Coefficients, \
                            Constant, VectorConstant, TensorConstant

# Split function
from ufl.split_functions import split

# Literal constants
from ufl.constantvalue import PermutationSymbol, Identity, zero, as_ufl

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
                       outer, inner, dot, cross, perp, \
                       det, inv, cofac, \
                       transpose, tr, diag, diag_vector, \
                       dev, skew, sym, \
                       sqrt, exp, ln, erf, \
                       cos, sin, tan, \
                       acos, asin, atan, atan_2, \
                       cosh, sinh, tanh, \
                       bessel_J, bessel_Y, bessel_I, bessel_K, \
                       eq, ne, le, ge, lt, gt, And, Or, Not, \
                       conditional, sign, Max, Min, \
                       variable, diff, \
                       Dx,  grad, div, curl, rot, nabla_grad, nabla_div, Dn, exterior_derivative, \
                       jump, avg, cell_avg, facet_avg, \
                       elem_mult, elem_div, elem_pow, elem_op

# Form class
from ufl.form import Form

# Integral classes
from ufl.integral import Integral, Measure, register_domain_type, ProductMeasure

# Representations of transformed forms
from ufl.formoperators import replace, derivative, action, energy_norm, rhs, lhs,\
    system, functional, adjoint, sensitivity_rhs #, dirichlet_functional

# Predefined convenience objects
from ufl.objects import \
    vertex, interval, triangle, tetrahedron, \
    quadrilateral, hexahedron, facet, cell1D, cell2D, cell3D, \
    i, j, k, l, p, q, r, s, \
    dx, ds, dS, dP, dE, dc

# Useful constants
from math import e, pi

__all__ = [
    'product',
    'get_handler', 'get_logger', 'set_handler', 'set_level', 'add_logfile',
    'UFLException', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
    'Cell', 'ProductCell',
    'SpatialCoordinate', 'FacetNormal',
    'CellVolume', 'Circumradius', 'CellSurfaceArea',
    'FacetArea', 'MinFacetEdgeLength', 'MaxFacetEdgeLength', 'FacetDiameter',
    'LocalCoordinate', 'GeometryJacobi',
    'GeometryJacobiDeterminant', 'InverseGeometryJacobi',
    'Domain', 'Region',
    'FiniteElementBase', 'FiniteElement',
    'MixedElement', 'VectorElement', 'TensorElement', 'EnrichedElement',
    'RestrictedElement', 'TensorProductElement',
    'register_element', 'show_elements',
    'Argument', 'TestFunction', 'TrialFunction',
    'Arguments', 'TestFunctions', 'TrialFunctions',
    'Coefficient', 'Coefficients',
    'Constant', 'VectorConstant', 'TensorConstant',
    'split',
    'PermutationSymbol', 'Identity', 'zero', 'as_ufl',
    'Index', 'indices',
    'as_tensor', 'as_vector', 'as_matrix', 'relabel',
    'unit_vector', 'unit_vectors', 'unit_matrix', 'unit_matrices',
    'rank', 'shape',
    'outer', 'inner', 'dot', 'cross', 'perp',
    'det', 'inv', 'cofac',
    'transpose', 'tr', 'diag', 'diag_vector', 'dev', 'skew', 'sym',
    'sqrt', 'exp', 'ln', 'erf',
    'cos', 'sin', 'tan',
    'acos', 'asin', 'atan', 'atan_2',
    'cosh', 'sinh', 'tanh',
    'bessel_J', 'bessel_Y', 'bessel_I', 'bessel_K',
    'eq', 'ne', 'le', 'ge', 'lt', 'gt', 'And', 'Or', 'Not',
    'conditional', 'sign', 'Max', 'Min',
    'variable', 'diff',
    'Dx', 'grad', 'div', 'curl', 'rot', 'nabla_grad', 'nabla_div', 'Dn', 'exterior_derivative',
    'jump', 'avg', 'cell_avg', 'facet_avg',
    'elem_mult', 'elem_div', 'elem_pow', 'elem_op',
    'Form',
    'Integral', 'Measure', 'register_domain_type', 'ProductMeasure',
    'replace', 'derivative', 'action', 'energy_norm', 'rhs', 'lhs',
    'system', 'functional', 'adjoint', 'sensitivity_rhs',
    'dx', 'ds', 'dS', 'dP', 'dE', 'dc',
    'vertex', 'interval', 'triangle', 'tetrahedron',
    'quadrilateral', 'hexahedron', 'facet',
    'cell1D', 'cell2D', 'cell3D',
    'i', 'j', 'k', 'l', 'p', 'q', 'r', 's',
    'e', 'pi',
    ]
