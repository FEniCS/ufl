"""This module defines evaluation algorithms for converting
converting UFL expressions to swiginac representation."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-22 -- 2008-08-26"

from collections import defaultdict

from ..common import some_key, product, Stack, StackDict
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

import swiginac as sw


class Context:
    "Context class for obtaining terminal expressions."
    def __init__(self):
        pass
    
    def x(self, component):
        return NotImplemented
    
    def facet_normal(self, component):
        return NotImplemented
        
    def function(self, i, component):
        return NotImplemented
    
    def basisfunction(self, i, component):
        return NotImplemented
    
    def variable(self, i, component, index2value, return_expression):
        if return_expression:
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



class SwiginacEvaluator(object):
    "Algorithm for evaluation of an UFL expression as a swiginac expression."
    def __init__(self, context):
        self._context = context
        
        self._components = Stack()
        self._index2value = StackDict()
        
        self._pre_handlers = {}
        self._post_handlers = {}

        ### Pre:
        h = self._pre_handlers
        
        # Terminals:
        h[Number] = self.h_number
        h[Variable] = self.h_variable
        h[BasisFunction] = self.h_basisfunction
        h[Function] = self.h_function
        h[Constant] = self.h_function
        h[FacetNormal] = self.h_facet_normal
        
        # Repeated index objects:
        h[Product] = self.h_product
        h[Indexed] = self.h_indexed
        h[Tensor] = self.h_tensor
        h[PartialDerivative] = self.h_partial_derivative
        h[Diff] = self.h_diff
 
        ### Post:
        h = self._post_handlers
        
        # Basic algebra:
        h[Sum] = self.h_sum
        h[Division] = self.h_division
        h[Power] = self.h_power
        #h[Mod] = self.h_mod
        h[Abs]  = self.h_abs
        
        # MathFunctions:
        h[Sqrt] = self.h_sqrt
        h[Exp]  = self.h_exp
        h[Ln]   = self.h_ln
        h[Cos]  = self.h_cos
        h[Sin]  = self.h_sin
        
        # Containers:
        h[ListVector] = self.h_list_vector
        h[ListMatrix] = self.h_list_matrix
        
        # Discontinuous operators:
        h[PositiveRestricted] = self.h_positive_restricted
        h[NegativeRestricted] = self.h_negative_restricted

    def component(self):
        "Return current component tuple."
        if len(self._components):
            return self._components.peek()
        return ()

    def transform(self, expression):
        "Transform a subexpression in the current context."
        c = expression.__class__
        
        # Case 1: Convert children first: handler will take their transformed expressions as input
        handler = self._post_handlers.get(c, None)
        if handler is not None:
            ops = tuple(self.transform(o) for o in expression.operands())
            return handler(expression, *ops)
        
        # Case 2: Do not convert children: handler will take care of its own children
        handler = self._pre_handlers.get(c, None)
        if handler is not None:
            return handler(expression)
        
        ufl_error("No handler implemented for class %s" % c)

    ### Handlers for basic terminal objects:
    
    def h_number(self, x):
        return sw.numeric(x._value)
    
    def h_variable(self, x):
        # Lookup variable and evaluate its expression directly if not found
        v = self._context.variable(x._count, self.component(), self._index2value) # FIXME: symbol flag
        if v is None:
            v = self.transform(x._expression)
        return v

    def h_basisfunction(self, x):
        ufl_assert(len(self._index2value) == 0, "Shouldn't have any indices left to map at this point!")
        return self._context.basisfunction(x._count, self.component())

    def h_function(self, x):
        ufl_assert(len(self._index2value) == 0, "Shouldn't have any indices left to map at this point!")
        return self._context.function(x._count, self.component())
    
    def h_facet_normal(self, x):
        ufl_assert(len(self._index2value) == 0, "Shouldn't have any indices left to map at this point!")
        return self._context.facet_normal(self.component())
    
    ### Handlers for basic algebra:
    
    def h_sum(self, x, *ops):
        return sum(ops)

    def h_product(self, x):
        ops = x.operands()
        ri = x._repeated_indices
        if ri:
            # FIXME: Handle repeated indices
            ufl_error("Not implemented")
            for i in ri: # FIXME
                somevalue = 0 # FIXME
                self._index2value.push(i, somevalue)
                ops = [self.transform(o) for o in ops]
                self._index2value.pop()
            return sum(FIXME)
        else:
            ops = [self.transform(o) for o in ops]
            return product(ops)

    def h_division(self, x, a, b):
        return a / b

    def h_power(self, x, a, b):
        return a ** b

    def h_mod(self, x, a, b):
        ufl_error("Mod not supported by swiginac.")
        return a % b

    def h_abs(self, x, a):
        return abs(a)

    ### Basic math functions:
    def h_sqrt(self, x, y):
        return sw.sqrt(y)
    
    def h_exp(self, x, y):
        return sw.exp(y)
    
    def h_ln(self, x, y):
        return sw.ln(y)
    
    def h_cos(self, x, y):
        return sw.cos(y)
    
    def h_sin(self, x, y):
        return sw.sin(y)
    
    ### Index handling: 
    
    def h_indexed(self, x, A, ii):
        ops = x.operands()
        ri = x._repeated_indices
        
        # Assertions about original UFL object x
        ufl_assert(ops[0].shape(), "Expecting tensor with some shape.")
        ufl_assert(ops[1] is ii, "Expecting unchanged MultiIndex in h_indexed.")

        # Assertions on handled objects A and ii:
        ufl_assert(isinstance(ii, MultiIndex), "Expecting unchanged MultiIndex in h_indexed.")

        # FIXME: What do we get in, what do we return, what's in between?
        ufl_assert(isinstance(A, Variable), \
            "Expecting indexed expression to be wrapped in a Variable (a temporary implementation issue).")

        # FIXME: update indices here, evt. in a summation loop for repeated indices
        self._index2value.push(FIXME)
        component = tuple(self.component())
        B = self._context.variable(A._count, component, self._index2value) 
        # FIXME: restore indices here
        self._index2value.pop()

        # For each index in ii._indices:
        # - FixedIndex: take a slice of A (constant index in one dimension)
        # - Axis: take a slice of A (":" in one dimension)
        # - Index:
        #     - Among repeated: Sum slices of A
        #     - Free: Maybe we should have as input a mapping "I: Index -> int" and treat like FixedIndex.
        #             If we can place variables at suitable places first, we can still avoid recomputations.

        return x
    
    ### Container handling:
    
    def h_list_vector(self, x, *ops):
        component = self.component()
        ufl_assert(len(components) == 1, "Got %d indices for a list component." % len(component))
        i = component[0]
        ufl_assert(isinstance(i, int), "Can't index list with %s." % repr(i))
        return ops[i]
    
    def h_list_matrix(self, x, *ops):
        component = self.component()
        ufl_assert(len(component) == 2, "Got %d indices for a matrix." % len(component))
        i = component[0]
        j = component[1]
        ufl_assert(isinstance(i, int), "Can't index matrix row with %s." % repr(i))
        ufl_assert(isinstance(j, int), "Can't index matrix column with %s." % repr(j))
        return ops[i][j]
    
    def h_tensor(self, x, *ops):
        component = tuple(self.component())
        for i, idx in enumerate(component):
            self._index2value.push((ops[1][i], idx))
        result = self.transform(ops[0])
        for i in range(len(component)):
            component.pop()
        return result
    
    ### Differentiation:
    # We cannot just use sw.diff on the expression f since it
    # may depend on symbols refering to functions depending on y.
    # If we assume that AD has been applied for derivatives,
    # then the expression to differentiate is always a terminal.
    # TODO: Need algorithm to apply AD to all kinds of derivatives!
    def h_partial_derivative(self, x, f, y):
        ufl_error("Not implemented")
        ufl_assert(isinstance(x, Terminal) or (isinstance(x, Variable) and isinstance(x._expression, Terminal)), \
            "Expecting to differentiate a Terminal object, you must apply AD first!")
        
        # TODO: Apply indices to get the right part of f, it may be a tensor!
        # TODO: Is y a MultiIndex here? Pick the right symbol to diff with from context!
        #ufl_assert(isinstance(y, sw.symbol), "Expecting a swiginac.symbol to differentate w.r.t.")
        return sw.diff(f, y)
    
    def h_diff(self, x, f, y):
        ufl_error("Not implemented")
        ufl_assert(isinstance(x, Terminal), "Expecting to differentiate a Terminal object, you must apply AD first!")
        
        # TODO: Apply indices to get the right part of f and y, they may be tensors!
        ufl_assert(isinstance(y, sw.symbol), "Expecting a swiginac.symbol to differentate w.r.t.")
        return sw.diff(f, y)
    
    ### Interior facet stuff:
    
    def h_positive_restricted(self, x, *ops):
        ufl_error("Not implemented")
        return FIXME
    
    def h_negative_restricted(self, x, *ops):
        ufl_error("Not implemented")
        return FIXME
 

   
### Requires code structure and thus shouldn't occur in SwiginacEvaluator
# (i.e. any conditionals should be wrapped in variables before coming here) # TODO
#d[EQ] = self.h_EQ
#d[NE] = self.h_NE
#d[LE] = self.h_LE
#d[GE] = self.h_GE
#d[LT] = self.h_LT
#d[GT] = self.h_GT
#d[Conditional] = self.h_Conditional

### Replaced by expand_compounds:
#d[Identity] = self.h_Identity
#d[Transposed] = self.h_Transposed
#d[Outer] = self.h_Outer
#d[Inner] = self.h_Inner
#d[Dot] = self.h_Dot
#d[Cross] = self.h_Cross
#d[Trace] = self.h_Trace
#d[Determinant] = self.h_Determinant
#d[Inverse] = self.h_Inverse
#d[Deviatoric] = self.h_Deviatoric
#d[Cofactor] = self.h_Cofactor
#d[Grad] = self.h_Grad
#d[Div] = self.h_Div
#d[Curl] = self.h_Curl
#d[Rot] = self.h_Rot


def expression2swiginac(expression, context):
    s = SwiginacEvaluator(context)
    return s.transform(expression)


def form2swiginac(form): # TODO: Move to SFC
    # TODO: transform form with form transformations, renumberings, expands, dependency splits etc.
    # TODO: get UFL basisfunctions etc from form
    # TODO: build basic context with all coefficients and geometry
    context = Context() # FIXME
    s = SwiginacEvaluator(context)
    
    # for all variables in variable stacks independent of basisfunctions: evaluate variables
    # for all variables in variable stacks depending on a single basisfunction: update context and evaluate variables
    # for all variables in variable stacks depending on multiple basisfunction: update context and evaluate variables
    # for all basisfunctions, update context

    e = s.transform(expression, d)

    return e

