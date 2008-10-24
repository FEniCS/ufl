"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-15 -- 2008-10-24"

# Modified by Anders Logg, 2008

from .base import UFLObject, Terminal, FloatValue, ZeroType
from .variable import Variable
from .finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from .basisfunction import BasisFunction, TestFunction, TrialFunction
from .function import Function, VectorConstant, TensorConstant, Constant
from .geometry import FacetNormal
from .indexing import MultiIndex, Indexed, Index, FixedIndex, AxisType
#from .indexing import AxisType, as_index, as_index_tuple, extract_indices
from .tensors import ListTensor, ComponentTensor
from .algebra import Sum, Product, Division, Power, Abs
from .tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from .mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from .restriction import Restricted, PositiveRestricted, NegativeRestricted
from .differentiation import SpatialDerivative, Diff, Grad, Div, Curl, Rot
from .conditional import Condition, EQ, NE, LE, GE, LT, GT, Conditional
from .form import Form
from .integral import Integral

# Make sure we import baseoperators which attaches special functions to UFLObject
from . import baseoperators as __baseoperators

# Collect all classes in lists
__classobj = type(UFLObject)
def __issubclass(x, y):
    return isinstance(x, __classobj) and issubclass(x, y)
abstract_classes    = set([UFLObject, Terminal, Restricted, Condition, MathFunction])
all_ufl_classes     = set([c for c in locals().values() if __issubclass(c, UFLObject)])
ufl_classes         = set([c for c in all_ufl_classes if c not in abstract_classes])
terminal_classes    = set([c for c in ufl_classes if __issubclass(c, Terminal)])
nonterminal_classes = set([c for c in ufl_classes if not __issubclass(c, Terminal)])

# Add _uflid to all classes:
for c in all_ufl_classes:
    c._uflid = c
