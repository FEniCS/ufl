"""Utility algorithms for inspection of and information extraction from UFL objects in various ways."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-24"

# Modified by Anders Logg, 2008

from itertools import chain

from ..output import ufl_assert, ufl_error
from ..common import lstr

from ..base import UFLObject, Terminal
from ..algebra import Sum, Product, Division
from ..basisfunction import BasisFunction
from ..function import Function
from ..variable import Variable
from ..function import Function, Constant
from ..tensors import ListTensor, ComponentTensor
from ..tensoralgebra import Transposed, Inner, Dot, Outer, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor, Skew
from ..restriction import PositiveRestricted, NegativeRestricted
from ..differentiation import SpatialDerivative, Diff, Grad, Div, Curl, Rot
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ..indexing import DefaultDimType, Indexed
from ..form import Form
from ..integral import Integral
from ..classes import terminal_classes, nonterminal_classes
from .traversal import iter_expressions, post_traversal


#--- Utilities to extract information from an expression ---

def extract_type(a, ufl_type):
    """Build a set of all objects of class ufl_type found in a.
    The argument a can be a Form, Integral or UFLObject."""
    iter = (o for e in iter_expressions(a) \
              for (o, stack) in post_traversal(e) \
              if isinstance(o, ufl_type) )
    return set(iter)

def extract_classes(a):
    """Build a set of all unique UFLObject subclasses used in a.
    The argument a can be a Form, Integral or UFLObject."""
    c = set()
    for e in iter_expressions(a):
        for (o, stack) in post_traversal(e):
            c.add(o._uflid) #c.add(type(o))
    return c

def extract_domain(a):
    "Find the polygonal domain of Form a."
    element = extract_elements(a)
    domain = element[0].domain()
    return domain

def extract_value_shape(expression, dimension):
    "Evaluate the value shape of expression with given implicit dimension."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFL expression.")
    ufl_assert(isinstance(dimension, int), "Expecting int dimension.")
    s = expression.shape()
    shape = []
    for i in s:
        if isinstance(i, DefaultDimType):
            shape.append(dimension)
        else:
            shape.append(i)
    return tuple(shape)

def extract_basisfunctions(a):
    """Build a sorted list of all basisfunctions in a,
    which can be a Form, Integral or UFLObject."""
    # build set of all unique basisfunctions
    s = extract_type(a, BasisFunction)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject."""
    # build set of all unique coefficients
    s = extract_type(a, Function)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

# alternative implementation, kept as an example:
def _extract_coefficients(a):
    """Build a sorted list of all coefficients in a,
    which can be a Form, Integral or UFLObject."""
    # build set of all unique coefficients
    s = set()
    def func(o):
        if isinstance(o, Function):
            s.add(o)
    walk(a, func)
    # sort by count
    l = sorted(s, cmp=lambda x,y: cmp(x._count, y._count))
    return l

def extract_elements(a):
    "Build a sorted list of all elements used in a."
    return tuple(f._element for f in chain(extract_basisfunctions(a), extract_coefficients(a)))

def extract_unique_elements(a):
    "Build a set of all unique elements used in a."
    return set(extract_elements(a))

def extract_variables(a):
    """Build a set of all Variable objects in a,
    which can be a Form, Integral or UFLObject."""
    return extract_type(a, Variable)

def extract_indices(expression):
    "Build a set of all Index objects used in expression."
    multi_indices = extract_type(expression, MultiIndex)
    indices = set()
    for mi in multi_indices:
        indices.update(i for i in mi if isinstance(i, Index))
    return indices

def extract_duplications(expression):
    "Build a set of all repeated expressions in expression."
    ufl_assert(isinstance(expression, UFLObject), "Expecting UFL expression.")
    handled = set()
    duplicated = set()
    for (o, stack) in post_traversal(expression):
        if o in handled:
            duplicated.add(o)
        handled.add(o)
    return duplicated

def extract_monomials(expression, indent=""):
    "Extract monomial representation of expression (if possible)."

    # FIXME: Not yet working, need to include derivatives, integrals etc

    ufl_assert(isinstance(expression, Form) or isinstance(expression, UFLObject), "Expecting UFL form or expression.")

    # Iterate over expressions
    m = []

    print ""
    print "Extracting monomials"

    #cell_integrals = expression.cell_integrals()
    #print cell_integrals
    #print dir(cell_integrals[0].)
    #integrals

    for e in iter_expressions(expression):

        # Check for linearity
        if not e.is_linear():
            ufl_error("Operator is nonlinear, unable to extract monomials: " + str(e))
            
        print indent + "e =", e, str(type(e))
        operands = e.operands()
        if isinstance(e, Sum):
            ufl_assert(len(operands) == 2, "Strange, expecting two terms.")
            m += extract_monomials(operands[0], indent + "  ")
            m += extract_monomials(operands[1], indent + "  ")
        elif isinstance(e, Product):
            ufl_assert(len(operands) == 2, "Strange, expecting two factors.")
            for m0 in extract_monomials(operands[0], indent + "  "):
                for m1 in extract_monomials(operands[1], indent + "  "):
                    m.append(m0 + m1)
        elif isinstance(e, BasisFunction):
            m.append((e,))
        elif isinstance(e, Function):
            m.append((e,))
        else:
            print type(e)
            print e.as_basic()
            print "free indices =", e.free_indices()
            ufl_error("Don't know how to handle expression: %s", str(e))

    return m


def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    c = expression._uflid #c = type(expression)
    if c in handlers:
        h = handlers[c]
    else:
        ufl_error("Didn't find class %s among handlers." % c)
    return h(expression, *ops)

class NotMultiLinearException(Exception):
    pass

def extract_basisfunction_dependencies(expression):
    "TODO: Document me."
    h = {}
    
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
    h[Diff] = h_diff

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

