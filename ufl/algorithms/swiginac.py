"""This module defines evaluation algorithms for converting
converting UFL expressions to swiginac representation."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2008-08-22"

from collections import defaultdict

from ..common import some_key, product
from ..output import ufl_assert, ufl_error

# All classes:
from ..base import UFLObject, Terminal, Number
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunctions import BasisFunction, Function, Constant
#from ..basisfunctions import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index
#from ..indexing import FixedIndex, AxisType, as_index, as_index_tuple, extract_indices
from ..tensors import ListVector, ListMatrix, Tensor
#from ..tensors import Vector, Matrix
from ..algebra import Sum, Product, Division, Power, Mod, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral
#from ..formoperators import Derivative, Action, Rhs, Lhs # TODO: What to do with these?

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients, indices


class Context:
    "Context class for obtaining terminal expressions."
    def basisfunction(self, i):
        return NotImplemented
    
    def coefficient(self, i):
        return NotImplemented
    
    def n(self):
        return NotImplemented
    

def evaluate_as_swiginac(expression, context):
    
    ops = expression.operands()
    for o in ops:
        pass
    
    
    