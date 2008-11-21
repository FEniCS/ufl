"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-15 -- 2008-11-21"

# Modified by Anders Logg, 2008

from ufl.base import Expr, Terminal
from ufl.zero import Zero
from ufl.scalar import ScalarValue, FloatValue, IntValue, ScalarSomething
from ufl.variable import Variable
from ufl.finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ufl.basisfunction import BasisFunction, TestFunction, TrialFunction
from ufl.function import Function, VectorConstant, TensorConstant, Constant
from ufl.geometry import FacetNormal
from ufl.indexing import MultiIndex, Indexed, Index, FixedIndex, AxisType
from ufl.tensors import ListTensor, ComponentTensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Cofactor, Inverse, Deviatoric, Skew
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import Condition, EQ, NE, LE, GE, LT, GT, Conditional
from ufl.form import Form
from ufl.integral import Integral

# Make sure we import baseoperators which attaches special functions to Expr
from ufl import baseoperators as __baseoperators

# Collect all classes in lists
__classobj = type(Expr)
def __issubclass(x, y):
    return isinstance(x, __classobj) and issubclass(x, y)
abstract_classes    = set([Expr, Terminal, Restricted, Condition, MathFunction])
all_ufl_classes     = set([c for c in locals().values() if __issubclass(c, Expr)])
ufl_classes         = set([c for c in all_ufl_classes if c not in abstract_classes])
terminal_classes    = set([c for c in ufl_classes if __issubclass(c, Terminal)])
nonterminal_classes = set([c for c in ufl_classes if not __issubclass(c, Terminal)])

# Add _uflid to all classes:
for c in all_ufl_classes:
    c._uflid = c
