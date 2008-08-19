"""This file is useful for external code like tests and form compilers,
since it enables the syntax "from ufl.classes import FooBar" for getting
implementation details not exposed through the default ufl namespace."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-15 -- 2008-08-19"

from .base import UFLObject, Terminal, Number, Compound
from .variable import Variable
from .finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from .basisfunctions import BasisFunction, Function, Constant
#from .basisfunctions import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from .geometry import FacetNormal
from .indexing import MultiIndex, Indexed
#from .indexing import Index, FixedIndex, AxisType, as_index, as_index_tuple, extract_indices
from .tensors import ListVector, ListMatrix, Tensor
#from .tensors import Vector, Matrix
from .algebra import Sum, Product, Division, Power, Mod, Abs
from .tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from .mathfunctions import MathFunction
from .restriction import Restricted, PositiveRestricted, NegativeRestricted
from .differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot
from .form import Form
from .integral import Integral
from .formoperators import Derivative, Action, Rhs, Lhs

# Make sure we import baseoperators which attaches special functions to UFLObject
from . import baseoperators as __baseoperators

# Collect all classes in lists
ufl_classes = [c for c in locals().values() if isinstance(c, UFLObject)]
terminal_classes = [c for c in ufl_classes if isinstance(c, Terminal)]
nonterminal_classes = [c for c in ufl_classes if not isinstance(c, Terminal)]
