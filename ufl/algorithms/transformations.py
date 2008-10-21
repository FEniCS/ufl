"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-21"

from collections import defaultdict
from itertools import izip

from ..common import some_key, product
from ..output import ufl_assert, ufl_error, ufl_warning
from ..permutation import compute_indices

# FIXME: Lots of imports duplicated in all algorithm modules
# FIXME: Should be cleaned up


#- Implement lhs, rhs, and is_multilinear:
#   - Start with helper function
#        def basisfunction_combinations(expression):
#   - Implement is_multilinear based on this
#   - lhs and rhs should be trivial now in the case of 
#     forms where the splitting happen at the integral
#     level or in a root-level sum in the integrand.
#     Cases like "(f + 3*u)*v*dx" needs more involved computation.


# All classes:
from ..base import UFLObject, Terminal, FloatValue, ZeroType
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunction import BasisFunction
#from ..basisfunction import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ..function import Function, Constant
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index, FixedIndex
#from ..indexing import AxisType, as_index, as_index_tuple, extract_indices
from ..tensors import ListTensor, ComponentTensor
from ..algebra import Sum, Product, Division, Power, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import SpatialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..form import Form
from ..integral import Integral

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients, indices, duplications

def transform_integrands(a, transformation):
    """Transform all integrands in a form with a transformation function.
    
    Example usage:
      b = transform_integrands(a, flatten)
    """
    ufl_assert(isinstance(a, Form), "Expecting a Form.")
    integrals = []
    for itg in a._integrals:
        integrand = transformation(itg._integrand)
        newitg = Integral(itg._domain_type, itg._domain_id, integrand)
        integrals.append(newitg)
    
    return Form(integrals)


def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    c = type(expression)
    if c in handlers:
        h = handlers[c]
    else:
        ufl_error("Didn't find class %s among handlers." % c)
    return h(expression, *ops)


def ufl_reuse_handlers():
    """This function constructs a handler dict for transform
    which can be used to reconstruct a ufl expression through
    transform(...). Nonterminal objects are reused if possible."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in ufl_reuse_handlers. Add to classes.py." % type(x))
    def make_not_implemented():
        return not_implemented
    d = defaultdict(make_not_implemented)
    # Terminal objects are simply reused:
    def this(x):
        return x
    for c in terminal_classes:
        d[c] = this
    # Non-terminal objects are reused if all their children are untouched
    def reconstruct(x, *ops):
        if all((a is b) for (a,b) in izip(x.operands(), ops)):
            return x
        else:
            return type(x)(*ops)
    for c in nonterminal_classes:
        d[c] = reconstruct
    return d


def ufl_copy_handlers():
    """This function constructs a handler dict for transform
    which can be used to reconstruct a ufl expression through
    transform(...). Nonterminal objects are copied, such that 
    no nonterminal objects are shared between the new and old
    expression."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in ufl_copy_handlers. Add to classes.py." % type(x))
    def make_not_implemented():
        return not_implemented
    d = defaultdict(make_not_implemented)
    # Terminal objects are simply reused:
    def this(x):
        return x
    for c in terminal_classes:
        d[c] = this
    # Non-terminal objects are reused if all their children are untouched
    def reconstruct(x, *ops):
        return type(x)(*ops)
    for c in nonterminal_classes:
        d[c] = reconstruct
    return d


def ufl2ufl(expression):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    handlers = ufl_reuse_handlers()
    return transform(expression, handlers)


def ufl2uflcopy(expression):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    handlers = ufl_copy_handlers()
    return transform(expression, handlers)


def expand_compounds(expression, dim):
    """Convert an UFL expression to a new UFL expression, with all 
    compound operator objects converted to basic (indexed) expressions."""
    d = ufl_reuse_handlers()
    def e_compound(x, *ops):
        return x.as_basic(dim, *ops)
    for c in compound_classes:
        d[c] = e_compound
    return transform(expression, d)


def flatten(expression):
    """Convert an UFL expression to a new UFL expression, with sums 
    and products flattened from binary tree nodes to n-ary tree nodes."""
    d = ufl_reuse_handlers()
    def _flatten(x, *ops):
        c = type(x)
        newops = []
        for o in ops:
            if isinstance(o, c):
                newops.extend(o.operands())
            else:
                newops.append(o)
        return c(*newops)
    d[Sum] = _flatten
    d[Product] = _flatten
    return transform(expression, d)


def replace(expression, substitution_map):
    """Replace objects in expression.
    
    @param expression:
        A UFLObject.
    @param substitution_map:
        A dict with from:to replacements to perform.
    """
    handlers = ufl_reuse_handlers()
    orig_handlers = {}
    def r_replace(x, *ops):
        y = substitution_map.get(x)
        if y is None:
            return orig_handlers[type(x)](x, *ops)
        return y
    for k in substitution_map.keys():
        c = type(k)
        orig_handlers[c] = handlers[c]
        handlers[c] = r_replace
    return transform(expression, handlers)


def replace_in_form(form, substitution_map):
    "Apply replace to all integrands in a form."
    def replace_expression(expression):
        return replace(expression, substitution_map)
    return transform_integrands(form, replace_expression)


# TODO: Take care when using this, it will replace _all_ occurences of these indices,
# so in f.ex. (a[i]*b[i]*(1.0 + c[i]*d[i]) the replacement i==0 will result in
# (a[0]*b[0]*(1.0 + c[0]*d[0]) which is probably not what is wanted.
# If this is a problem, a new algorithm may be needed.

