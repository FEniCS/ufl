"""This module defines evaluation algorithms for converting
converting UFL expressions to swiginac representation."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2008-08-23"

from collections import defaultdict

from ..common import some_key, product, StackDict
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
    def __init__(self):
        self._components = []
        def none():
            return None
        self._index_value_map = defaultdict(none)
    
    def push_component(self, component):
        self._components.append(component)

    def get_component(self):
        return self._components[-1]

    def pop_component(self):
        return self._components.pop()
        
    def x(self, component):
        return NotImplemented
    
    def facet_normal(self, component):
        return NotImplemented
        
    def function(self, i, component):
        return NotImplemented
    
    def basisfunction(self, i, component):
        return NotImplemented
    
    def variable(self, i, component, index_value_map):
        if self.allow_symbol:
            return None
        return None


# FIXME: Implement for all UFL expressions!
# Steps:
# - Implement outer controller which knows about code structure and variable lists (context object is defined by this controller)
# - Algorithm to insert variables in Indexed objects (or do in splitting)
# - Handle indexed tensor expressions (assuming expand_compounds has been applied)
#   Need more special cases in transform!
# - Handle derivatives d/ds and d/dx with AD (need this for quadrature code)
# - Handle conditionals (cannot express with swiginac, need to represent code structure)



def transform(expression, pre_handlers, post_handlers, context):
    """..."""
    c = expression.__class__
    handler = pre_handlers[c]
    if handler is None:
        # Convert children first: handler will take their transformed expressions as input
        ops = tuple(transform(o, pre_handlers, post_handlers, context) for o in expression.operands())
        handler = post_handlers[c]
        return handler(expression, context, *ops)
    # Do not convert children: handler will take care of its own children
    return handler(expression, context)


def evaluate_as_swiginac(expression, context):
    index_value_map = StackDict()
    pre_handlers, post_handlers = swiginac_handlers(context, index_value_map)
    return transform(expression, pre_handlers, post_handlers, context)


def swiginac_handlers(context, index_value_map):
    sw = swiginac
    
    # Show a clear error message if we miss some types here:
    def not_implemented(x, ops):
        ufl_error("No handler defined for %s in swiginac_handlers." % x.__class__)
    d = defaultdict(not_implemented)
    
    pre_handlers, post_handlers # FIXME: Use these instead of d
    
    ### Basic terminal objects:
    def s_number(x):
        return sw.numeric(x._value)
    d[Number] = s_number
    
    def s_variable(x):
        # Lookup variable and evaluate its expression directly if not found
        v = context.variable(x._count, component, index_value_map) 
        if v is None:
            return evaluate_as_swiginac(x._expression, context, index_value_map, component)
        return v
    d[Variable] = s_variable

    def s_basisfunction(x):
        ufl_assert(len(index_value_map) == 0, "Shouldn't have any indices left to map at this point!")
        return context.basisfunction(x._count, component)
    d[BasisFunction] = s_basisfunction

    def s_function(x):
        ufl_assert(len(index_value_map) == 0, "Shouldn't have any indices left to map at this point!")
        return context.function(x._count, component)
    d[Function] = s_function
    d[Constant] = s_function

    def s_facet_normal(x):
        ufl_assert(len(index_value_map) == 0, "Shouldn't have any indices left to map at this point!")
        return context.facet_normal(component)
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
        ops = x.operands()
        ri = x._repeated_indices
        
        # Assertions about original UFL object x
        ufl_assert(ops[0].shape(), "Expecting tensor with some shape.")
        ufl_assert(ops[1] is ii, "Expecting unchanged MultiIndex in s_indexed.")

        # Assertions on handled objects A and ii:
        ufl_assert(isinstance(ii, MultiIndex), "Expecting unchanged MultiIndex in s_indexed.")

        # FIXME: What do we get in, what do we return, what's in between?
        ufl_assert(isinstance(A, Variable), "Expecting indexed expression to be wrapped in a Variable (a temporary implementation issue).")

        # FIXME: update indices here, evt. in a summation loop for repeated indices
        index_value_map.push(FIXME)
        component = FIXME
        B = context.variable(A._count, component, index_value_map) 
        # FIXME: restore indices here
        index_value_map.pop()

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
        ufl_assert(len(component) == 1, "Got %d indices for a list component." % len(component))
        i = component[0]
        ufl_assert(isinstance(i, int), "Can't index list with %s." % repr(i))
        return ops[i]
    d[ListVector] = s_list_vector
    
    def s_list_matrix(x, *ops):
        ufl_assert(len(component) == 2, "Got %d indices for a matrix." % len(component))
        i = component[0]
        j = component[1]
        ufl_assert(isinstance(i, int), "Can't index matrix row with %s." % repr(i))
        ufl_assert(isinstance(j, int), "Can't index matrix column with %s." % repr(j))
        return ops[i][j]
    d[ListMatrix] = s_list_matrix
    
    def s_tensor(x, *ops):
        for i, idx in enumerate(component):
            index_value_map.push((ops[1][i], idx))
        result = evaluate_as_swiginac(ops[0], index_value_map) # FIXME: Must change algorithm, using conflicting traversal directions..
        for i in range(len(component)):
            component.pop()
        return result
    d[Tensor] = s_tensor
    
    ### Differentiation:
    # We cannot just use sw.diff on the expression f since it
    # may depend on symbols refering to functions depending on y.
    # If we assume that AD has been applied for derivatives,
    # then the expression to differentiate is always a terminal.
    # TODO: Need algorithm to apply AD to all kinds of derivatives!
    #def s_partial_derivative(x, f, y):
    #    ufl_assert(isinstance(x, Terminal) or (isinstance(x, Variable) and isinstance(x._expression, Terminal), \
    #        "Expecting to differentiate a Terminal object, you must apply AD first!")
    #    
    #    # TODO: Apply indices to get the right part of f, it may be a tensor!
    #    # TODO: Is y a MultiIndex here? Pick the right symbol to diff with from context!
    #    #ufl_assert(isinstance(y, sw.symbol), "Expecting a swiginac.symbol to differentate w.r.t.")
    #    return sw.diff(f, y)
    #d[PartialDerivative] = s_partial_derivative
    #
    #def s_diff(x, f, y):
    #    ufl_assert(isinstance(x, Terminal), "Expecting to differentiate a Terminal object, you must apply AD first!")
    #    
    #    # TODO: Apply indices to get the right part of f and y, they may be tensors!
    #    ufl_assert(isinstance(y, sw.symbol), "Expecting a swiginac.symbol to differentate w.r.t.")
    #    return sw.diff(f, y)
    #d[Diff] = s_diff
    
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
    
    return pre_handlers, post_handlers


def evaluate_form_as_swiginac(form):
    # transform form with form transformations, renumberings, expands, dependency splits etc.
    # get UFL basisfunctions etc from form
    # build basic context with all coefficients and geometry
    # for all variables in variable stacks independent of basisfunctions: evaluate variables
    # for all variables in variable stacks depending on a single basisfunction: update context and evaluate variables
    # for all variables in variable stacks depending on multiple basisfunction: update context and evaluate variables
    # for all basisfunctions, update context
    evaluate_as_swiginac(expression, context)
    d = swiginac_handlers(context)
    return transform(expression, d)

