"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-11-21"

# Modified by Anders Logg, 2008

from itertools import chain

from ufl.output import ufl_assert, ufl_error
from ufl.common import lstr, UFLTypeDefaultDict

from ufl.base import Expr, Terminal
from ufl.algebra import Sum, Product, Division
from ufl.basisfunction import BasisFunction
from ufl.function import Function
from ufl.variable import Variable
from ufl.function import Function, Constant
from ufl.tensors import ListTensor, ComponentTensor
from ufl.tensoralgebra import Transposed, Inner, Dot, Outer, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor, Skew
from ufl.restriction import PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative, Grad, Div, Curl, Rot
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ufl.indexing import Indexed, Index, MultiIndex
from ufl.form import Form
from ufl.integral import Integral
from ufl.classes import terminal_classes, nonterminal_classes
from ufl.algorithms.traversal import iter_expressions, post_traversal, walk


#--- Utilities to extract information from an expression ---

def extract_classes(a):
    """Build a set of all unique Expr subclasses used in a.
    The argument a can be a Form, Integral or Expr."""
    return set(o._uflid for e in iter_expressions(a) \
                        for (o, stack) in post_traversal(e))

def extract_type(a, ufl_type):
    """Build a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or Expr."""
    return set(o for e in iter_expressions(a) \
                 for (o, stack) in post_traversal(e) \
                 if isinstance(o, ufl_type))

def extract_terminals(a):
    "Build a set of all Terminal objects in a."
    return set(o for e in iter_expressions(a) \
                 for (o,stack) in post_traversal(e) \
                 if isinstance(o, Terminal))

def extract_basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or Expr."""
    def c(x,y):
        return cmp(x._count, y._count)
    return sorted(extract_type(a, BasisFunction), cmp=c)

def extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    def c(x,y):
        return cmp(x._count, y._count)
    return sorted(extract_type(a, Function), cmp=c)

# alternative implementation, kept as an example:
def _extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or Expr."""
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Function):
            s.add(o)
    walk(a, func)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x.count(), y.count()))
    return l

def extract_elements(a):
    "Build a sorted list of all elements used in a."
    args = chain(extract_basisfunctions(a), extract_coefficients(a))
    return tuple(f.element() for f in args)

def extract_unique_elements(a):
    "Build a set of all unique elements used in a."
    return set(extract_elements(a))

def extract_indices(expression):
    "Build a set of all Index objects used in expression."
    ufl_info("Is this used for anything? Doesn't make much sense.")
    multi_indices = extract_type(expression, MultiIndex)
    indices = set()
    for mi in multi_indices:
        indices.update(i for i in mi if isinstance(i, Index))
    return indices

def extract_variables(a):
    """Build a set of all Variable objects in a,
    which can be a Form, Integral or Expr."""
    return extract_type(a, Variable)
# FIXME: Does these two do the exact same thing?
def extract_variables(expression, handled_vars=None):
    if handled_vars is None:
        handled_vars = set()
    if isinstance(expression, Variable):
        i = expression._count
        if i in handled_vars:
            return []
        handled_vars.add(i)
        variables = list(extract_variables(expression._expression, handled_vars))
        variables.append(expression)
    else:
        variables = []
        for o in expression.operands():
            variables.extend(extract_variables(o, handled_vars))
    return variables

def extract_duplications(expression):
    "Build a set of all repeated expressions in expression."
    # TODO: Handle indices in a canonical way, maybe create a transformation that does this to apply before extract_duplications?
    ufl_assert(isinstance(expression, Expr), "Expecting UFL expression.")
    handled = set()
    duplicated = set()
    for (o, stack) in post_traversal(expression):
        if o in handled:
            duplicated.add(o)
        handled.add(o)
    return duplicated



def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    c = expression._uflid
    h = handlers.get(c, None)
    if c is None:
        ufl_error("Didn't find class %s among handlers." % c)
    return h(expression, *ops)

class NotMultiLinearException(Exception):
    pass

def extract_basisfunction_dependencies(expression): # TODO: Reimplement as a Transformer
    "TODO: Document me."
    def not_implemented(x, *ops):
        ufl_error("No handler implemented in extract_basisfunction_dependencies for '%s'" % str(x._uflid))
    h = UFLTypeDefaultDict(not_implemented)
    
    # Default for terminals: no dependency on basis functions 
    def h_terminal(x):
        return frozenset()
    for c in terminal_classes:
        h[c] = h_terminal
    
    def h_basisfunction(x):
        return frozenset((frozenset((x,)),))
    h[BasisFunction] = h_basisfunction
    
    # Default for nonterminals: nonlinear in all arguments 
    def h_nonlinear(x, *opdeps):
        for o in opdeps:
            if o: raise NotMultiLinearException, repr(x)
        return frozenset()
    for c in nonterminal_classes:
        h[c] = h_nonlinear
    
    # Some nonterminals are linear in their single argument 
    def h_linear(x, a):
        return a
    h[Grad] = h_linear
    h[Div] = h_linear
    h[Curl] = h_linear
    h[Rot] = h_linear
    h[Transposed] = h_linear
    h[Trace] = h_linear
    h[Skew] = h_linear
    h[PositiveRestricted] = h_linear
    h[NegativeRestricted] = h_linear

    def h_indexed(x, f, i):
        if i: raise NotMultiLinearException, repr(x)
        return f
    h[Indexed] = h_indexed
    
    def h_diff(x, a, b):
        if b: raise NotMultiLinearException, repr(x)
        return a
    h[SpatialDerivative] = h_diff
    h[VariableDerivative] = h_diff

    def h_variable(x):
        return extract_basisfunction_dependencies(x._expression)
    h[Variable] = h_variable

    def h_componenttensor(x, f, i):
        return f
    h[ComponentTensor] = h_componenttensor
    
    # Require same dependencies for all listtensor entries
    def h_listtensor(x, *opdeps):
        d = opdeps[0]
        for d2 in opdeps[1:]:
            if not d == d2:
                raise NotMultiLinearException, repr(x)
        return d
    h[ListTensor] = h_listtensor
    
    # Considering EQ, NE, LE, GE, LT, GT nonlinear in this context. 
    def h_conditional(x, cond, t, f):
        if cond or (not t == f):
            raise NotMultiLinearException, repr(x)
        return t

    # Basis functions cannot be in the denominator
    def h_division(x, a, b):
        if b: raise NotMultiLinearException, repr(x)
        return a
    h[Division] = h_division

    # Sums can contain both linear and bilinear terms (we could change this to require that all operands have the same dependencies)
    def h_sum(x, *opdeps):
        deps = set(opdeps[0])
        for o in opdeps[1:]:
            # o is here a set of sets
            deps |= o
        return frozenset(deps)
    h[Sum] = h_sum
    
    # Product operands should not depend on the same basis functions
    def h_product(x, *opdeps):
        c = []
        adeps, bdeps = opdeps # TODO: Generalize to any number of operands
        for ad in adeps:
            for bd in bdeps:
                cd = ad | bd
                if not len(cd) == len(ad) + len(bd):
                    raise NotMultiLinearException, repr(x)
                c.append(cd)
        return frozenset(c)
    h[Product] = h_product
    h[Inner] = h_product
    h[Outer] = h_product
    h[Dot] = h_product
    h[Cross] = h_product
    
    return transform(expression, h)

