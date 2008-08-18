"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-08-15"

from collections import defaultdict

from ..output import ufl_assert

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
from ..tensoralgebra import Identity, Transpose, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import PartialDerivative, Diff, DifferentialOperator, Grad, Div, Curl, Rot
from ..form import Form
from ..integral import Integral
from ..formoperators import Derivative, Action, Rhs, Lhs, rhs, lhs

# Other algorithms:
from .analysis import basisfunctions, coefficients


def criteria_not_argument(a):
    return not isinstance(a, (Function, BasisFunction))

def criteria_not_trial_function(a):
    return not (isinstance(a, BasisFunction) and (a._count > 0 or a._count == -1))

def criteria_not_basis_function(a):
    return not isinstance(a, BasisFunction)

def _detect_argument_dependencies(a, criteria):
    """Detect edges in expression tree where subtrees
    depend on different stages of form arguments.
    A Variable object is inserted at each edge.
    
    Stage 0:  Subtrees that does not depend on any arguments.
    Stage 1:  Subtrees that does not depend on any basisfunctions (i.e., that only depend on coefficients).
    Stage 2:  Subtrees that does not depend on basisfunction 1 (i.e. trial function for a matrix)
    Stage 3:  Subtrees that does not depend on basisfunction 0 (i.e. test function)
    """
    ufl_warning("NB! Assumes renumbered basisfunctions! FIXME: Implement basisfunction renumbering.")
    
    if isinstance(a, Terminal):
        return a, criteria(a)
    
    operands = []
    crit = []
    for o in a.operands():
        b, c = _detect_argument_dependencies(o, criteria)
        operands.append(b)
        crit.append(c)
    
    # FIXME: finish this
    
    if False:
        return a   
    return a.__class__(*operands)


def substitute_indices(u, indices, values):
    """Substitute Index objects from list indices with corresponding
    fixed values from list values in expression u."""
    ufl_error("Not implemented") # FIXME: Implement
    return u


def expand_indices(expression):
    "Expand implicit summations into explicit Sums of Products."
    
    # TODO: Could be a good idea to insert Variable instances around expanded sums.
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFLObject instance.")
    
    def e_product(x, *ops):
        # TODO: Find all repeated indices:
        ii = ()
        a = x
        return Product(*ops) # FIXME
    
    def e_pdiff(x, *ops):
        return PartialDerivative(*ops) # FIXME
    
    handlers = _ufl_handlers()
    handlers[Product] = e_product
    handlers[PartialDerivative] = e_pdiff
    
    # FIXME: Handle all other necessary objects here, f.ex. the Diff object h resulting from "g = variable(g); f = foo(g); h = diff(f[i], g[i])"
    
    return transform(expression, handlers)

