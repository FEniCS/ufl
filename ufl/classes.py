"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-15 -- 2009-01-23"

# Modified by Anders Logg, 2008

from ufl.expr import Expr
from ufl.terminal import Terminal, Tuple
from ufl.zero import Zero
from ufl.scalar import ScalarValue, FloatValue, IntValue, ScalarSomething
from ufl.variable import Variable, Label
from ufl.finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ufl.basisfunction import BasisFunction, TestFunction, TrialFunction
from ufl.function import Function, VectorConstant, TensorConstant, Constant
from ufl.geometry import SpatialCoordinate, FacetNormal
from ufl.indexing import MultiIndex, Indexed, IndexBase, Index, FixedIndex, IndexSum
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Cofactor, Inverse, Deviatoric, Skew
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.differentiation import Derivative, FunctionDerivative, SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import Condition, EQ, NE, LE, GE, LT, GT, Conditional
from ufl.form import Form
from ufl.integral import Integral

# Make sure we import exproperators which attaches special functions to Expr
from ufl import exproperators as __exproperators

# Collect all classes in lists
__all_classes       = (c for c in locals().values() if isinstance(c, type))
all_ufl_classes     = set(c for c in __all_classes if issubclass(c, Expr))
abstract_classes    = set((Expr, Terminal, Restricted, Condition, MathFunction)) # TODO: Can we extract this as well by looking at the subclasses of 
ufl_classes         = set(c for c in all_ufl_classes if c not in abstract_classes)
terminal_classes    = set(c for c in all_ufl_classes if issubclass(c, Terminal))
nonterminal_classes = set(c for c in all_ufl_classes if not issubclass(c, Terminal))

# Add _uflid to all classes:
for c in all_ufl_classes:
    c._uflid = c
