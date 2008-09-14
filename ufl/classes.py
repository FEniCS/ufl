"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-15 -- 2008-09-14"

# Modified by Anders Logg, 2008

from .base import UFLObject, Terminal, FloatValue, Compound
from .variable import Variable
from .finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from .basisfunction import BasisFunction, TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from .function import Function, Constant, Functions
from .geometry import FacetNormal
from .indexing import MultiIndex, Indexed
#from .indexing import Index, FixedIndex, AxisType, as_index, as_index_tuple, extract_indices
from .tensors import ListVector, ListMatrix, Tensor
#from .tensors import Vector, Matrix
from .algebra import Sum, Product, Division, Power, Mod, Abs
from .tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from .mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from .restriction import Restricted, PositiveRestricted, NegativeRestricted
from .differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot
from .conditional import Condition, EQ, NE, LE, GE, LT, GT, Conditional
from .form import Form
from .integral import Integral
from .formoperators import Derivative, Action, Rhs, Lhs

# Make sure we import baseoperators which attaches special functions to UFLObject
from . import baseoperators as __baseoperators

# Collect all classes in lists
__classobj = type(UFLObject)
def __issubclass(x, y):
    return isinstance(x, __classobj) and issubclass(x, y)
ufl_classes         = [c for c in locals().values() if __issubclass(c, UFLObject)]
terminal_classes    = [c for c in ufl_classes if __issubclass(c, Terminal)]
nonterminal_classes = [c for c in ufl_classes if not __issubclass(c, Terminal)]
compound_classes    = [c for c in ufl_classes if __issubclass(c, Compound)]
