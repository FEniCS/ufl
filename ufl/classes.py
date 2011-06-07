"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
#
# First added:  2008-08-15
# Last changed: 2009-12-08

from ufl.assertions import ufl_assert
from ufl.expr import Expr, Operator, WrapperType, AlgebraOperator
from ufl.terminal import Terminal, FormArgument, UtilityType, Tuple
from ufl.constantvalue import ConstantValue, Zero, ScalarValue, FloatValue, IntValue, Identity, PermutationSymbol
from ufl.variable import Variable, Label
from ufl.finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ufl.argument import Argument, TestFunction, TrialFunction
from ufl.coefficient import Coefficient, ConstantBase, VectorConstant, TensorConstant, Constant
from ufl.geometry import GeometricQuantity, SpatialCoordinate, FacetNormal, Space, Cell, CellVolume, Circumradius
from ufl.indexing import MultiIndex, Indexed, IndexBase, Index, FixedIndex, IndexSum
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import CompoundTensorOperator, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Cofactor, Inverse, Deviatoric, Skew, Sym
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin, Tan, Acos, Asin, Atan
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.lifting import LiftingResult, LiftingOperatorResult, LiftingFunctionResult, TerminalOperator, LiftingOperator, LiftingFunction
from ufl.differentiation import Derivative, CompoundDerivative, CoefficientDerivative, SpatialDerivative, VariableDerivative, Grad, Div, Curl
from ufl.conditional import Condition, EQ, NE, LE, GE, LT, GT, Conditional
from ufl.form import Form
from ufl.integral import Measure, Integral

# Make sure we import exproperators which attaches special functions to Expr
from ufl import exproperators as __exproperators

# Collect all classes in lists
__all_classes       = (c for c in locals().values() if isinstance(c, type))
all_ufl_classes     = set(c for c in __all_classes if issubclass(c, Expr))
abstract_classes    = set(s for c in all_ufl_classes for s in c.mro()[1:-1])
abstract_classes.remove(Coefficient)
ufl_classes         = set(c for c in all_ufl_classes if c not in abstract_classes)
terminal_classes    = set(c for c in all_ufl_classes if issubclass(c, Terminal))
nonterminal_classes = set(c for c in all_ufl_classes if not issubclass(c, Terminal))

# Add _uflclass and _classid to all classes:
from ufl.common import camel2underscore as _camel2underscore
for _i, _c in enumerate(all_ufl_classes):
    _c._classid = _i
    _c._uflclass = _c
    _c._handlername = _camel2underscore(_c.__name__)

#__all__ = all_ufl_classes
