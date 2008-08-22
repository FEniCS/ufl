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

import swiginac


class Context:
    "Context class for obtaining terminal expressions."
    def basisfunction(self, i):
        return NotImplemented
    
    def coefficient(self, i):
        return NotImplemented
    
    def facet_normal(self):
        return NotImplemented
    
    def variable(self, i):
        return None


# FIXME: Implement for all UFL expressions!
# Steps:
# - Handle indexed tensor expressions (assuming expand_compounds has been applied)
# - Handle derivatives d/ds and d/dx with AD (need this for quadrature code)
# - Implement outer controller which knows about code structure and variable lists
# - Handle conditionals (cannot express with swiginac, need to represent code structure)


def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    return handlers[expression.__class__](expression, *ops)


def swiginac_handlers(context):
    
    sw = swiginac

    # Show a clear error message if we miss some types here:
    def not_implemented(x, ops):
        ufl_error("No handler defined for %s in swiginac_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    
    ### Basic terminal objects:
    def s_number(x):
        return sw.numeric(x._value)
    d[Number] = s_number
    
    def s_variable(x):
        # Lookup variable and evaluate its expression directly if not found
        v = context.variable(x._count)
        if v is None:
            return evaluate_as_swiginac(x._expression, context)
        return v
    d[Variable] = s_variable

    def s_basisfunction(x):
        return context.basisfunction(x._count)
    d[BasisFunction] = s_basisfunction

    def s_function(x):
        return context.function(x._count)
    d[Function] = s_function
    d[Constant] = s_function

    def s_facet_normal(x):
        return context.facet_normal()
    d[FacetNormal] = s_facet_normal
    
    ### Basic algebra:
    def s_sum(x, *ops):
        return sum(ops)
    d[Sum] = s_sum

    def s_product(x, *ops):
        return product(ops)
    d[Product] = s_product

    def s_division(x, a, b):
        return a / b
    d[Division] = s_division

    def s_power(x, a, b):
        return a ** b
    d[Power] = s_power

    #def s_mod(x, a, b):
    #    return a % b
    #d[Mod] = s_mod

    def s_abs(x, a):
        return abs(a)
    d[Abs] = s_abs

    ### Basic math functions:
    def s_sqrt(x, y):
        return sw.sqrt(y)
    d[Sqrt] = s_sqrt
    
    def s_exp(x, y):
        return sw.exp(y)
    d[Exp] = s_exp
    
    def s_ln(x, y):
        return sw.ln(y)
    d[Ln] = s_ln
    
    def s_cos(x, y):
        return sw.cos(y)
    d[Cos] = s_cos
    
    def s_sin(x, y):
        return sw.sin(y)
    d[Sin] = s_sin
    
    ### Index handling: 
    def s_indexed(x, A, ii):
        # Assertions about original UFL object x
        ops = x.operands()
        ufl_assert(ops[0].shape(), "Expecting tensor with some shape.")
        ufl_assert(ops[1] is ii, "Expecting unchanged MultiIndex in s_indexed.")

        # Assertions on handled objects A and ii:
        ufl_assert(isinstance(ii, MultiIndex), "Expecting unchanged MultiIndex in s_indexed.")

        # FIXME: What do we get in, what do we return, what's in between?
        ufl_assert(isinstance(A, WHAT), "")
        ri = x._repeated_indices

        # For each index in ii._indices:
        # - FixedIndex: take a slice of A (constant index in one dimension)
        # - Axis: take a slice of A (":" in one dimension)
        # - Index:
        #     - Among repeated: Sum slices of A
        #     - Free: Maybe we should have as input a mapping "I: Index -> int" and treat like FixedIndex.
        #             If we can place variables at suitable places first, we can still avoid recomputations.

        return x
    d[Indexed] = s_indexed

    # MultiIndex can't be converted to swiginac,
    # keep as it is and handle in parent
    def s_multi_index(x):
        return x
    d[MultiIndex] = s_multi_index 

    ### Container handling:
    def s_list_vector(x, *ops):
        return TODO
    d[ListVector] = s_list_vector
    
    def s_list_matrix(x, *ops):
        return TODO
    d[ListMatrix] = s_list_matrix
    
    def s_tensor(x, *ops):
        return TODO
    d[Tensor] = s_tensor
    
    ### Differentiation:
    # FIXME: We cannot just use sw.diff on the subexpression
    # if that is a symbol or it depends on symbols.
    # If we assume that AD has been applied for derivatives,
    # then the expression to differentiate is always a terminal.
    def s_partial_derivative(x, f, y):
        return sw.diff(f, y) # FIXME
    d[PartialDerivative] = s_partial_derivative

    def s_diff(x, f, y):
        return sw.diff(f, y) # FIXME
    d[Diff] = s_diff
    
    ### Interior facet stuff:
    #def s_positive_restricted(x, *ops):
    #    return TODO
    #d[PositiveRestricted] = s_positive_restricted
    #def s_negative_restricted(x, *ops):
    #    return TODO
    #d[NegativeRestricted] = s_negative_restricted
    
    ### Requires code structure and thus shouldn't occur in here
    # (i.e. any conditionals should be wrapped in variables before coming here) # TODO
    #d[EQ] = s_EQ
    #d[NE] = s_NE
    #d[LE] = s_LE
    #d[GE] = s_GE
    #d[LT] = s_LT
    #d[GT] = s_GT
    #d[Conditional] = s_Conditional
    
    ### Replaced by expand_compounds:
    #d[Identity] = s_Identity
    #d[Transposed] = s_Transposed
    #d[Outer] = s_Outer
    #d[Inner] = s_Inner
    #d[Dot] = s_Dot
    #d[Cross] = s_Cross
    #d[Trace] = s_Trace
    #d[Determinant] = s_Determinant
    #d[Inverse] = s_Inverse
    #d[Deviatoric] = s_Deviatoric
    #d[Cofactor] = s_Cofactor
    #d[Grad] = s_Grad
    #d[Div] = s_Div
    #d[Curl] = s_Curl
    #d[Rot] = s_Rot

    return d


def evaluate_as_swiginac(expression, context):
    d = swiginac_handlers(context)
    return transform(expression, d)


def evaluate_form_as_swiginac(form):
    # transform form with form transformations, renumberings, expands, dependency splits etc.
    # get basisfunctions etc
    # build context
    # for all variables in 
    # for all basisfunctions, update context
    evaluate_as_swiginac(expression, context)
    d = swiginac_handlers(context)
    return transform(expression, d)

