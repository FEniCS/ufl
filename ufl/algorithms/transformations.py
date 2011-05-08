"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes and Anders Logg"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-05-07 -- 2009-11-17"

# Modified by Anders Logg, 2009-2010
# Last changed: 2010-10-25

from itertools import izip
from inspect import getargspec

from ufl.log import error, warning, debug, info
from ufl.common import Stack, StackDict
from ufl.assertions import ufl_assert
from ufl.finiteelement import TensorElement
from ufl.classes import Expr, Terminal, Product, Index, FixedIndex, ListTensor, Variable, Zero
from ufl.indexing import indices, complete_shape
from ufl.tensors import as_tensor, as_matrix, as_vector, ListTensor, ComponentTensor
from ufl.form import Form
from ufl.integral import Integral
from ufl.classes import all_ufl_classes
from ufl.algorithms.analysis import has_type, extract_duplications

def transform(expression, handlers):
    """Convert a UFLExpression according to rules defined by
    the mapping handlers = dict: class -> conversion function."""
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [transform(o, handlers) for o in expression.operands()]
    c = expression._uflclass
    h = handlers.get(c, None)
    if c is None:
        error("Didn't find class %s among handlers." % c)
    return h(expression, *ops)

class MultiFunction(object):
    """Base class for collections of nonrecursive expression node handlers."""
    _handlers_cache = {}
    def __init__(self):
        # Analyse class properties and cache handler data the
        # first time this is run for a particular class
        cache_data = MultiFunction._handlers_cache.get(type(self))
        if not cache_data:
            cache_data = [None]*len(all_ufl_classes)
            # For all UFL classes
            for classobject in all_ufl_classes:
                # Iterate over the inheritance chain (NB! This assumes that all UFL classes inherits a single Expr subclass and that this is the first superclass!)
                for c in classobject.mro():
                    # Register classobject with handler for the first encountered superclass
                    name = c._handlername
                    if getattr(self, name, None):
                        cache_data[classobject._classid] = name
                        break
            MultiFunction._handlers_cache[type(self)] = cache_data
        # Build handler list for this particular class (get functions bound to self)
        self._handlers = [getattr(self, name) for name in cache_data]

    def __call__(self, o, *args, **kwargs):
        h = self._handlers[o._classid]
        return h(o, *args, **kwargs)

def is_post_handler(function):
    "Is this a handler that expects transformed children as input?"
    insp = getargspec(function)
    num_args = len(insp[0]) + int(insp[1] is not None)
    visit_children_first = num_args > 2
    return visit_children_first

class Transformer(object):
    """Base class for a visitor-like algorithm design pattern used to
    transform expression trees from one representation to another."""
    _handlers_cache = {}
    def __init__(self, variable_cache=None):
        if variable_cache is None:
            variable_cache = {}
        self._variable_cache = variable_cache

        # Analyse class properties and cache handler data the
        # first time this is run for a particular class
        cache_data = Transformer._handlers_cache.get(type(self))
        if not cache_data:
            cache_data = [None]*len(all_ufl_classes)
            # For all UFL classes
            for classobject in all_ufl_classes:
                # Iterate over the inheritance chain (NB! This assumes that all UFL classes inherits a single Expr subclass and that this is the first superclass!)
                for c in classobject.mro():
                    # Register classobject with handler for the first encountered superclass
                    name = c._handlername
                    function = getattr(self, name, None)
                    if function:
                        cache_data[classobject._classid] = name, is_post_handler(function)
                        break
            Transformer._handlers_cache[type(self)] = cache_data

        # Build handler list for this particular class (get functions bound to self)
        self._handlers = [(getattr(self, name), post) for (name, post) in cache_data]

        # Keep a stack of objects visit is called on, to ease backtracking
        self._visit_stack = []

    def print_visit_stack(self):
        print "/"*80
        print "Visit stack in Transformer:"
        def sstr(s):
            ss = str(type(s)) + " ; "
            n = 80 - len(ss)
            return ss + str(s)[:n]
        print "\n".join(sstr(s) for s in self._visit_stack)
        print "\\"*80

    def visit(self, o):
        #debug("Visiting object of type %s." % type(o).__name__)
        # Update stack
        self._visit_stack.append(o)

        # Get handler for the UFL class of o (type(o) may be an external subclass of the actual UFL class)
        h, visit_children_first = self._handlers[o._classid]

        #if not h:
        #    # Failed to find a handler! Should never happen, but will happen if a non-Expr object is visited.
        #    error("Can't handle objects of type %s" % str(type(o)))

        # Is this a handler that expects transformed children as input?
        if visit_children_first:
            # Yes, visit all children first and then call h.
            r = h(o, *map(self.visit, o.operands()))
        else:
            # No, this is a handler that handles its own children
            # (arguments self and o, where self is already bound)
            r = h(o)

        # Update stack and return
        self._visit_stack.pop()
        return r

    def undefined(self, o):
        "Trigger error."
        error("No handler defined for %s." % o._uflclass.__name__)

    def reuse(self, o):
        "Always reuse Expr (ignore children)"
        return o

    def reuse_if_possible(self, o, *operands):
        "Reuse Expr if possible, otherwise reconstruct from given operands."
        #if all(a is b for (a, b) in izip(operands, o.operands())):
        if operands == o.operands():
            return o
        #return o.reconstruct(*operands)
        # Debugging version:
        try:
            r = o.reconstruct(*operands)
        except:
            print
            print
            print
            print "FAILURE in reuse_if_possible:"
            print "type(o) =", type(o)
            print "operands ="
            print
            print "\n\n".join(map(str,operands))
            print
            print "stack ="
            self.print_visit_stack()
            print
            raise
        return r

    def always_reconstruct(self, o, *operands):
        "Always reconstruct expr."
        return o.reconstruct(*operands)

    # Set default behaviour for any Expr
    expr = undefined

    # Set default behaviour for any Terminal
    terminal = reuse

    def reuse_variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        v = self._variable_cache.get(l)
        if v is not None:
            return v

        # Visit the expression our variable represents
        e2 = self.visit(e)

        # If the expression is the same, reuse Variable object
        if e == e2:
            v = o
        else:
            # Recreate Variable (with same label)
            v = Variable(e2, l)

        # Cache variable
        self._variable_cache[l] = v
        return v

    def reconstruct_variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        v = self._variable_cache.get(l)
        if v is not None:
            return v

        # Visit the expression our variable represents
        e2 = self.visit(e)

        # Always reconstruct Variable (with same label)
        v = Variable(e2, l)
        self._variable_cache[l] = v
        return v

class ReuseTransformer(Transformer):
    def __init__(self, variable_cache=None):
        Transformer.__init__(self, variable_cache)

    # Set default behaviour for any Expr
    expr = Transformer.reuse_if_possible

    # Set default behaviour for any Terminal
    terminal = Transformer.reuse

    # Set default behaviour for Variable
    variable = Transformer.reuse_variable

class CopyTransformer(Transformer):
    def __init__(self, variable_cache=None):
        Transformer.__init__(self, variable_cache)

    # Set default behaviour for any Expr
    expr = Transformer.always_reconstruct

    # Set default behaviour for any Terminal
    terminal = Transformer.reuse

    # Set default behaviour for Variable
    variable = Transformer.reconstruct_variable

class Replacer(ReuseTransformer):
    def __init__(self, mapping):
        ReuseTransformer.__init__(self)
        self._mapping = mapping
        ufl_assert(all(isinstance(k, Terminal) for k in mapping.keys()), \
            "This implementation can only replace Terminal objects.")

    def terminal(self, o):
        e = self._mapping.get(o)
        return o if e is None else e

class TreeFlattener(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)

    def sum_or_product(self, o, *ops):
        c = o._uflclass
        operands = []
        for b in ops:
            if isinstance(b, c):
                operands.extend(b.operands())
            else:
                operands.append(b)
        return o.reconstruct(*operands)
    sum = sum_or_product
    product = sum_or_product

class VariableStripper(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)

    def variable(self, o):
        return self.visit(o._expression)

#class OperatorApplier(ReuseTransformer):
#    "Implements mappings that can be defined through Python operators."
#    def __init__(self):
#        ReuseTransformer.__init__(self)
#
#    def abs(self, o, a):
#        return abs(a)
#
#    def sum(self, o, *ops):
#        return sum(ops)
#
#    def division(self, o, a, b):
#        return a / b
#
#    def power(self, o, a, b):
#        return a ** b
#
#    def product(self, o, *ops):
#        return product(ops)
#
#    def indexed(self, o, a, b):
#        return a[*b] if isinstance(b, tuple) else a[b]


# TODO: Indices will often mess up extract_duplications / mark_duplications.
# Can we renumber indices consistently from the leaves to avoid that problem?
# This may introduce many ComponentTensor/Indexed objects for relabeling of indices though.
# We probably need some kind of pattern matching to make this effective.
# That's another step towards a complete symbolic library...
#
# What this does do well is insert Variables around subexpressions that the
# user actually identified manually in his code like in "a = ...; b = a*(1+a)",
# and expressions without indices (prior to expand_compounds).
class DuplicationMarker(ReuseTransformer):
    def __init__(self, duplications):
        ReuseTransformer.__init__(self)
        self._duplications = duplications
        self._expr2variable = {}

    def expr(self, o, *ops):
        v = self._expr2variable.get(o)
        if v is None:
            oo = o
            # reconstruct if necessary
            if not ops == o.operands():
                o = o._uflclass(*ops)

            if (oo in self._duplications) or (o in self._duplications):
                v = Variable(o)
                self._expr2variable[o] = v
                self._expr2variable[oo] = v
            else:
                v = o
        return v

    def wrap_terminal(self, o):
        v = self._expr2variable.get(o)
        if v is None:
            if o in self._duplications:
                v = Variable(o)
                self._expr2variable[o] = v
            else:
                v = o
        return v
    argument = wrap_terminal
    coefficient = wrap_terminal
    constant = wrap_terminal
    facet_normal = wrap_terminal

    def variable(self, o):
        e, l = o.operands()
        v = self._expr2variable.get(e)
        if v is None:
            e2 = self.visit(e)
            # Unwrap expression from the newly created Variable wrapper
            # unless the original expression was a Variable, in which
            # case we possibly need to keep the label for correctness.
            if (not isinstance(e, Variable)) and isinstance(e2, Variable):
                e2 = e2._expression
            v = self._expr2variable.get(e2)
            if v is None:
                v = Variable(e2, l)
                self._expr2variable[e] = v
                self._expr2variable[e2] = v
        return v

# Note: To avoid typing errors, the expressions for cofactor and
# deviatoric parts below were created with the script
# tensoralgebrastrings.py under sandbox/scripts/

class CompoundExpander(ReuseTransformer):
    "Expands compound expressions to equivalent representations using basic operators."
    def __init__(self, geometric_dimension):
        ReuseTransformer.__init__(self)
        self._dim = geometric_dimension

        #if self._dim is None:
        #    warning("Got None for dimension, some compounds cannot be expanded.")

    # ------------ Compound tensor operators

    def trace(self, o, A):
        i = Index()
        return A[i,i]

    def transposed(self, o, A):
        i, j = indices(2)
        return as_tensor(A[i, j], (j, i))

    def _square_matrix_shape(self, A):
        sh = A.shape()
        if self._dim is not None:
            sh = complete_shape(sh, self._dim)
        ufl_assert(sh[0] == sh[1], "Expecting square matrix.")
        ufl_assert(sh[0] is not None, "Unknown dimension.")
        return sh

    def deviatoric(self, o, A):
        sh = self._square_matrix_shape(A)
        if sh[0] == 2:
            return as_matrix([[-1/2*A[1,1]+1/2*A[0,0],A[0,1]],[A[1,0],1/2*A[1,1]-1/2*A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([[-1/3*A[1,1]-1/3*A[2,2]+2/3*A[0,0],A[0,1],A[0,2]],[A[1,0],2/3*A[1,1]-1/3*A[2,2]-1/3*A[0,0],A[1,2]],[A[2,0],A[2,1],-1/3*A[1,1]+2/3*A[2,2]-1/3*A[0,0]]])
        error("dev(A) not implemented for dimension %s." % sh[0])

    def skew(self, o, A):
        i, j = indices(2)
        return as_matrix( (A[i,j] - A[j,i]) / 2, (i,j) )

    def sym(self, o, A):
        i, j = indices(2)
        return as_matrix( (A[i,j] + A[j,i]) / 2, (i,j) )

    def cross(self, o, a, b):
        def c(i, j):
            return Product(a[i],b[j]) - Product(a[j],b[i])
        return as_vector((c(1,2), c(2,0), c(0,1)))

    def dot(self, o, a, b):
        ai = indices(a.rank()-1)
        bi = indices(b.rank()-1)
        k  = indices(1)
        # Create an IndexSum over a Product
        s = a[ai+k]*b[k+bi]
        return as_tensor(s, ai+bi)

    def inner(self, o, a, b):
        ufl_assert(a.rank() == b.rank())
        ii = indices(a.rank())
        # Create multiple IndexSums over a Product
        s = a[ii]*b[ii]
        return s

    def outer(self, o, a, b):
        ii = indices(a.rank())
        jj = indices(b.rank())
        # Create a Product with no shared indices
        s = a[ii]*b[jj]
        return as_tensor(s, ii+jj)

    def determinant(self, o, A):
        sh = self._square_matrix_shape(A)

        def det2D(B, i, j, k, l):
            return B[i,k]*B[j,l]-B[i,l]*B[j,k]

        if len(sh) == 0:
            return A
        if sh[0] == 2:
            return det2D(A, 0, 1, 0, 1)
        if sh[0] == 3:
            # TODO: Verify signs in this expression
            return A[0,0]*det2D(A, 1, 2, 1, 2) + \
                   A[0,1]*det2D(A, 1, 2, 2, 0) + \
                   A[0,2]*det2D(A, 1, 2, 0, 1)
        # TODO: Implement generally for all dimensions?
        error("Determinant not implemented for dimension %d." % self._dim)

    def cofactor(self, o, A):
        sh = self._square_matrix_shape(A)

        if sh[0] == 2:
            return as_matrix([[A[1,1], -A[0,1]], [-A[1,0], A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([ \
                [ A[2,2]*A[1,1] - A[1,2]*A[2,1],
                 -A[0,1]*A[2,2] + A[0,2]*A[2,1],
                  A[0,1]*A[1,2] - A[0,2]*A[1,1]],
                [-A[2,2]*A[1,0] + A[1,2]*A[2,0],
                 -A[0,2]*A[2,0] + A[2,2]*A[0,0],
                  A[0,2]*A[1,0] - A[1,2]*A[0,0]],
                [ A[1,0]*A[2,1] - A[2,0]*A[1,1],
                  A[0,1]*A[2,0] - A[0,0]*A[2,1],
                  A[0,0]*A[1,1] - A[0,1]*A[1,0]] \
                ])
        elif sh[0] == 4:
            # TODO: Find common subexpressions here.
            # TODO: Better implementation?
            return as_matrix([ \
                [-A[3,3]*A[2,1]*A[1,2] + A[1,2]*A[3,1]*A[2,3] + A[1,1]*A[3,3]*A[2,2] - A[3,1]*A[2,2]*A[1,3] + A[2,1]*A[1,3]*A[3,2] - A[1,1]*A[3,2]*A[2,3],
                 -A[3,1]*A[0,2]*A[2,3] + A[0,1]*A[3,2]*A[2,3] - A[0,3]*A[2,1]*A[3,2] + A[3,3]*A[2,1]*A[0,2] - A[3,3]*A[0,1]*A[2,2] + A[0,3]*A[3,1]*A[2,2],
                  A[3,1]*A[1,3]*A[0,2] + A[1,1]*A[0,3]*A[3,2] - A[0,3]*A[1,2]*A[3,1] - A[0,1]*A[1,3]*A[3,2] + A[3,3]*A[1,2]*A[0,1] - A[1,1]*A[3,3]*A[0,2],
                  A[1,1]*A[0,2]*A[2,3] - A[2,1]*A[1,3]*A[0,2] + A[0,3]*A[2,1]*A[1,2] - A[1,2]*A[0,1]*A[2,3] - A[1,1]*A[0,3]*A[2,2] + A[0,1]*A[2,2]*A[1,3]],
                [ A[3,3]*A[1,2]*A[2,0] - A[3,0]*A[1,2]*A[2,3] + A[1,0]*A[3,2]*A[2,3] - A[3,3]*A[1,0]*A[2,2] - A[1,3]*A[3,2]*A[2,0] + A[3,0]*A[2,2]*A[1,3],
                  A[0,3]*A[3,2]*A[2,0] - A[0,3]*A[3,0]*A[2,2] + A[3,3]*A[0,0]*A[2,2] + A[3,0]*A[0,2]*A[2,3] - A[0,0]*A[3,2]*A[2,3] - A[3,3]*A[0,2]*A[2,0],
                 -A[3,3]*A[0,0]*A[1,2] + A[0,0]*A[1,3]*A[3,2] - A[3,0]*A[1,3]*A[0,2] + A[3,3]*A[1,0]*A[0,2] + A[0,3]*A[3,0]*A[1,2] - A[0,3]*A[1,0]*A[3,2],
                  A[0,3]*A[1,0]*A[2,2] + A[1,3]*A[0,2]*A[2,0] - A[0,0]*A[2,2]*A[1,3] - A[0,3]*A[1,2]*A[2,0] + A[0,0]*A[1,2]*A[2,3] - A[1,0]*A[0,2]*A[2,3]],
                [ A[3,1]*A[1,3]*A[2,0] + A[3,3]*A[2,1]*A[1,0] + A[1,1]*A[3,0]*A[2,3] - A[1,0]*A[3,1]*A[2,3] - A[3,0]*A[2,1]*A[1,3] - A[1,1]*A[3,3]*A[2,0],
                  A[3,3]*A[0,1]*A[2,0] - A[3,3]*A[0,0]*A[2,1] - A[0,3]*A[3,1]*A[2,0] - A[3,0]*A[0,1]*A[2,3] + A[0,0]*A[3,1]*A[2,3] + A[0,3]*A[3,0]*A[2,1],
                 -A[0,0]*A[3,1]*A[1,3] + A[0,3]*A[1,0]*A[3,1] - A[3,3]*A[1,0]*A[0,1] + A[1,1]*A[3,3]*A[0,0] - A[1,1]*A[0,3]*A[3,0] + A[3,0]*A[0,1]*A[1,3],
                  A[0,0]*A[2,1]*A[1,3] + A[1,0]*A[0,1]*A[2,3] - A[0,3]*A[2,1]*A[1,0] + A[1,1]*A[0,3]*A[2,0] - A[1,1]*A[0,0]*A[2,3] - A[0,1]*A[1,3]*A[2,0]],
                [-A[1,2]*A[3,1]*A[2,0] - A[2,1]*A[1,0]*A[3,2] + A[3,0]*A[2,1]*A[1,2] - A[1,1]*A[3,0]*A[2,2] + A[1,0]*A[3,1]*A[2,2] + A[1,1]*A[3,2]*A[2,0],
                 -A[3,0]*A[2,1]*A[0,2] - A[0,1]*A[3,2]*A[2,0] + A[3,1]*A[0,2]*A[2,0] - A[0,0]*A[3,1]*A[2,2] + A[3,0]*A[0,1]*A[2,2] + A[0,0]*A[2,1]*A[3,2],
                  A[0,0]*A[1,2]*A[3,1] - A[1,0]*A[3,1]*A[0,2] + A[1,1]*A[3,0]*A[0,2] + A[1,0]*A[0,1]*A[3,2] - A[3,0]*A[1,2]*A[0,1] - A[1,1]*A[0,0]*A[3,2],
                 -A[1,1]*A[0,2]*A[2,0] + A[2,1]*A[1,0]*A[0,2] + A[1,2]*A[0,1]*A[2,0] + A[1,1]*A[0,0]*A[2,2] - A[1,0]*A[0,1]*A[2,2] - A[0,0]*A[2,1]*A[1,2]] \
                ])
        error("Cofactor not implemented for dimension %s." % sh[0])

    def inverse(self, o, A):
        if A.rank() == 0:
            return 1.0 / A
        return self.cofactor(None, A) / self.determinant(None, A)  # TODO: Verify this expression. Ainv = Acofac / detA

    # ------------ Compound differential operators

    def div(self, o, a):
        i = Index()
        g = a[i] if a.rank() == 1 else a[...,i]
        return g.dx(i)

    def grad(self, o, a):
        jj = Index()
        if a.rank() > 0:
            ii = tuple(indices(a.rank()))
            return as_tensor(a[ii].dx(jj), tuple(ii + (jj,)))
        return as_tensor(a.dx(jj), (jj,))

    def curl(self, o, a):
        # o = curl a = "[a.dx(1), -a.dx(0)]"            if a.shape() == ()
        # o = curl a = "cross(nabla, (a0, a1, 0))[2]" if a.shape() == (2,)
        # o = curl a = "cross(nabla, a)"              if a.shape() == (3,)
        def c(i, j):
            return a[j].dx(i) - a[i].dx(j)
        sh = a.shape()
        if sh == ():
            return as_vector((a.dx(1), -a.dx(0)))
        if sh == (2,):
            return c(0,1)
        if sh == (3,):
            return as_vector((c(1,2), c(2,0), c(0,1)))
        error("Invalid shape %s of curl argument." % (sh,))

class NotMultiLinearException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class ArgumentDependencyExtracter(Transformer):
    def __init__(self):
        Transformer.__init__(self)
        self._empty = frozenset()

    def expr(self, o, *opdeps):
        "Default for nonterminals: nonlinear in all operands."
        for d in opdeps:
            if d:
                raise NotMultiLinearException, repr(o)
        return self._empty

    def terminal(self, o):
        "Default for terminals: no dependency on Arguments."
        return self._empty

    def variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        d = self._variable_cache.get(l)
        if d is None:
            # Visit the expression our variable represents
            d = self.visit(e)
            self._variable_cache[l] = d
        return d

    def argument(self, o):
        d = frozenset((o,))
        return frozenset((d,))

    def linear(self, o, a):
        "Nonterminals that are linear with a single argument."
        return a
    grad = linear
    div = linear
    curl = linear
    transposed = linear
    trace = linear
    skew = linear
    positive_restricted = linear
    negative_restricted = linear

    def indexed(self, o, f, i):
        return f

    def spatial_derivative(self, o, a, b):
        return a

    def variable_derivative(self, o, a, b):
        if b:
            raise NotMultiLinearException, repr(o)
        return a

    def component_tensor(self, o, f, i):
        return f

    def list_tensor(self, o, *opdeps):
        "Require same dependencies for all listtensor entries."
        d = opdeps[0]
        for d2 in opdeps[1:]:
            if not d == d2:
                raise NotMultiLinearException, repr(o)
        return d

    def conditional(self, o, cond, t, f):
        "Considering EQ, NE, LE, GE, LT, GT nonlinear in this context."
        if cond or (not t == f):
            raise NotMultiLinearException, repr(o)
        return t

    def division(self, o, a, b):
        "Basis functions cannot be in the denominator."
        if b:
            raise NotMultiLinearException, repr(o)
        return a

    def index_sum(self, o, f, i):
        "Index sums inherit the dependencies of their summand."
        return f

    def sum(self, o, *opdeps):
        """Sums can contain both linear and bilinear terms (we could change
        this to require that all operands have the same dependencies)."""
        # convert frozenset to a mutable set
        deps = set(opdeps[0])
        for d in opdeps[1:]:
            # d is a frozenset of frozensets
            deps.update(d)
        return frozenset(deps)

    # Product operands should not depend on the same basis functions
    def product(self, o, *opdeps):
        c = []
        adeps, bdeps = opdeps # TODO: Generalize to any number of operands using permutations
        # for each frozenset ad in the frozenset adeps
        ufl_assert(isinstance(adeps, frozenset), "Type error")
        ufl_assert(isinstance(bdeps, frozenset), "Type error")
        ufl_assert(all(isinstance(ad, frozenset) for ad in adeps), "Type error")
        ufl_assert(all(isinstance(bd, frozenset) for bd in bdeps), "Type error")
        none = frozenset((None,))
        noneset = frozenset((none,))
        if not adeps:
            adeps = noneset
        if not bdeps:
            bdeps = noneset
        for ad in adeps:
            # for each frozenset bd in the frozenset bdeps
            for bd in bdeps:
                # build frozenset cd with the combined Argument dependencies from ad and bd
                cd = (ad | bd) - none
                # build frozenset cd with the combined Argument dependencies from ad and bd
                if not len(cd) == len(ad - none) + len(bd - none):
                    raise NotMultiLinearException, repr(o)
                # remember this dependency combination
                if cd:
                    c.append(cd)
        return frozenset(c)
    inner = product
    outer = product
    dot = product
    cross = product

class DuplicationPurger(ReuseTransformer):
    "Replace all duplicated nodes from an UFL Expr."
    def __init__(self):
        ReuseTransformer.__init__(self)
        self._handled = {}
        #self._duplications = set()

    def expr(self, x, *ops):
        # Check cache
        e = self._handled.get(x)
        if e is None:
            # Reuse or reconstruct
            if ops == x.operands():
                e = x
            else:
                e = x._uflclass(*ops)
            # Update cache
            self._handled[x] = e
        #else:
        #    self._duplications.add(e)
        assert repr(x) == repr(e)
        return e

    def terminal(self, x):
        e = self._handled.get(x)
        if e is None:
            # Reuse
            e = x
            # Update cache
            self._handled[x] = e
        #else:
        #    self._duplications.add(e)
        return e

class SumDegreeEstimator(Transformer):

    def __init__(self, default_degree):
        Transformer.__init__(self)
        self.default_degree = default_degree

    def terminal(self, v):
        return 0

    def expr(self, v, *ops):
        return max(ops)

    def form_argument(self, v):
        return v.element().degree()

    def spatial_derivative(self, v, f, i):
        return max(f - 1, 0)

    def product(self, v, *ops):
        degrees = [op for op in ops if not op is None]
        nones = [op for op in ops if op is None]
        return sum(degrees) + self.default_degree*len(nones)

    def power(self, v, a, b):
        f, g = v.operands()
        try:
            gi = int(g)
            return a*gi
        except:
            pass
        # Something to a non-integer power... TODO: How to handle this?
        if b > 0:
            return a*b
        return a

class MaxDegreeEstimator(Transformer):

    def __init__(self, default_degree):
        Transformer.__init__(self)
        self.default_degree = default_degree

    def terminal(self, v):
        return 0

    def expr(self, v, *ops):
        return max(ops)

    def form_argument(self, v):
        return v.element().degree()

    #def spatial_derivative(self, v, f, i):
    #    return max(f - 1, 0)

    def product(self, v, *ops):
        degrees = [op for op in ops if not op is None]
        nones = [op for op in ops if op is None]
        return max(degrees + [self.default_degree])

#--- User interface functions ---

def transform_integrands(form, transform):
    """Apply transform(expression) to each integrand
    expression in form, or to form if it is an Expr."""

    if isinstance(form, Form):
        newintegrals = []
        for itg in form.integrals():
            integrand = transform(itg.integrand())
            if not isinstance(integrand, Zero):
                newitg = itg.reconstruct(integrand)
                newintegrals.append(newitg)
        if not newintegrals:
            debug("No integrals left after transformation, returning empty form.")
        return Form(newintegrals)

    elif isinstance(form, Integral):
        integral = form
        integrand = transform(integral.integrand())
        new_integral = integral.reconstruct(integrand)
        return new_integral

    elif isinstance(form, Expr):
        expr = form
        return transform(expr)
    else:
        error("Expecting Form or Expr.")

def apply_transformer(e, transformer):
    """Apply transformer.visit(expression) to each integrand
    expression in form, or to form if it is an Expr."""
    def _transform(expr):
        return transformer.visit(expr)
    return transform_integrands(e, _transform)

def ufl2ufl(e):
    """Convert an UFL expression to a new UFL expression, with no changes.
    This is used for testing that objects in the expression behave as expected."""
    return apply_transformer(e, ReuseTransformer())

def ufl2uflcopy(e):
    """Convert an UFL expression to a new UFL expression.
    All nonterminal object instances are replaced with identical
    copies, while terminal objects are kept. This is used for
    testing that objects in the expression behave as expected."""
    return apply_transformer(e, CopyTransformer())

def replace(e, mapping):
    """Replace terminal objects in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    return apply_transformer(e, Replacer(mapping))

def flatten(e): # TODO: Fix or remove! Maybe this works better now with IndexSum marking implicit summations.
    """Convert an UFL expression to a new UFL expression, with sums
    and products flattened from binary tree nodes to n-ary tree nodes."""
    warning("flatten doesn't work correctly for some indexed products, like (u[i]*v[i])*(q[i]*r[i])")
    return apply_transformer(e, TreeFlattener())

def expand_compounds(e, dim=None):
    """Expand compound objects into basic operators.
    Requires e to have a well defined domain,
    for the geometric dimension to be defined."""
    if dim is None:
        cell = e.cell()
        if cell is not None:
            ufl_assert(not cell.is_undefined(), "Cannot infer dimension from undefined cell.")
            dim = cell.geometric_dimension()
    return apply_transformer(e, CompoundExpander(dim))

def strip_variables(e):
    "Replace all Variable instances with the expression they represent."
    return apply_transformer(e, VariableStripper())

def mark_duplications(e):
    """Wrap subexpressions that are equal
    (completely equal, not mathematically equivalent)
    in Variable objects to facilitate subexpression reuse."""
    duplications = extract_duplications(e)
    return apply_transformer(e, DuplicationMarker(duplications))

def purge_duplications(e):
    """Replace any subexpressions in expression that
    occur more than once with a single instance."""
    return apply_transformer(e, DuplicationPurger())

def extract_basis_function_dependencies(e): # FIXME: Rename
    "Extract a set of sets of Arguments."
    ufl_assert(isinstance(e, Expr), "Expecting an Expr.")
    return ArgumentDependencyExtracter().visit(e)

def estimate_max_polynomial_degree(e, default_degree=1):
    """Estimate the maximum polymomial degree of all functions in the
    expression. For coefficients defined on an element with
    unspecified degree (None), the degree is set to the given default
    degree."""
    de = MaxDegreeEstimator(default_degree)
    if isinstance(e, Form):
        degrees = [de.visit(integral.integrand()) for integral in e.integrals()]
    elif isinstance(e, Integral):
        degrees = [de.visit(e.integrand())]
    else:
        degrees = [de.visit(e)]
    return max(degrees)

def estimate_total_polynomial_degree(e, default_degree=1):
    """Estimate total polynomial degree of integrand. For coefficients
    defined on an element with unspecified degree (None), the degree
    is set to the given default degree."""
    de = SumDegreeEstimator(default_degree)
    if isinstance(e, Form):
        degrees = [de.visit(integral.integrand()) for integral in e.integrals()]
    elif isinstance(e, Integral):
        degrees = [de.visit(e.integrand())]
    else:
        degrees = [de.visit(e)]
    return max(degrees)
