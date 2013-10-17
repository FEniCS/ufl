"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace.
It also contains functionality used by algorithms for dealing with groups
of classes, and for mapping types to different handler functions."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Modified by Anders Logg, 2009.
# Modified by Kristian B. Oelgaard, 2011
#
# First added:  2008-08-15
# Last changed: 2013-03-29

from ufl.assertions import ufl_assert

# Elements
from ufl.finiteelement import (FiniteElementBase, FiniteElement,
    MixedElement, VectorElement, TensorElement,
    EnrichedElement, RestrictedElement, TensorProductElement)

# Base class for all expressions
from ufl.expr import Expr

# Terminal types
from ufl.terminal import Terminal, FormArgument, UtilityType, Data
from ufl.constantvalue import ConstantValue, IndexAnnotated, Zero, ScalarValue,\
    FloatValue, IntValue, Identity, PermutationSymbol
from ufl.argument import Argument, TestFunction, TrialFunction
from ufl.coefficient import (Coefficient, ConstantBase,
    VectorConstant, TensorConstant, Constant)
from ufl.geometry import (Cell, ProductCell,
    GeometricQuantity,
    SpatialCoordinate, FacetNormal,
    CellVolume, Circumradius, CellSurfaceArea,
    FacetArea, MinFacetEdgeLength, MaxFacetEdgeLength, FacetDiameter,
    LocalCoordinate, GeometryJacobi,
    GeometryJacobiDeterminant, InverseGeometryJacobi)
from ufl.indexing import IndexBase, FixedIndex, Index, MultiIndex

# Operator types
from ufl.operatorbase import Operator, WrapperType, AlgebraOperator, Tuple
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.variable import Variable, Label
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import CompoundTensorOperator, Transposed, Outer,\
    Inner, Dot, Cross, Trace, Determinant, Cofactor, Inverse, Deviatoric, Skew, Sym
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Erf,\
    Cos, Sin, Tan, Cosh, Sinh, Tanh, Acos, Asin, Atan, Atan2, \
    BesselFunction, BesselJ, BesselY, BesselI, BesselK
from ufl.differentiation import Derivative, CompoundDerivative, CoefficientDerivative,\
    VariableDerivative, Grad, Div, Curl, NablaGrad, NablaDiv
from ufl.conditional import Condition, BinaryCondition,\
    EQ, NE, LE, GE, LT, GT,\
    AndCondition, OrCondition, NotCondition, Conditional
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted, CellAvg, FacetAvg

# Higher level abstractions
from ufl.integral import Measure, ProductMeasure, Integral
from ufl.form import Form
from ufl.equation import Equation

# Make sure we import exproperators which attaches special functions to Expr
from ufl import exproperators as __exproperators

# Collect all classes in sets automatically classified by some properties
__all_classes       = (c for c in locals().values() if isinstance(c, type))
all_ufl_classes     = set(c for c in __all_classes if issubclass(c, Expr))
abstract_classes    = set(s for c in all_ufl_classes for s in c.mro()[1:-1])
abstract_classes.remove(Coefficient)
ufl_classes         = set(c for c in all_ufl_classes if c not in abstract_classes)
terminal_classes    = set(c for c in all_ufl_classes if issubclass(c, Terminal))
nonterminal_classes = set(c for c in all_ufl_classes if not issubclass(c, Terminal))

# Add _uflclass and _classid to all classes:
from ufl.common import camel2underscore as _camel2underscore
for _i, _c in enumerate(sorted(all_ufl_classes, key=lambda x:x.__name__)):
    _c._classid = _i
    _c._uflclass = _c
    _c._handlername = _camel2underscore(_c.__name__)

#__all__ = all_ufl_classes
