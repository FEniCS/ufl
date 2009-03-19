"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2009-03-12"

from itertools import izip
from inspect import getargspec

from ufl.log import error, warning, debug
from ufl.common import Stack, StackDict
from ufl.assertions import ufl_assert
from ufl.finiteelement import TensorElement
from ufl.classes import Expr, Terminal, Product, Index, FixedIndex, ListTensor, Variable, Function, Zero
from ufl.indexing import indices, complete_shape
from ufl.tensors import as_tensor, as_matrix, as_vector
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
    
    def visit(self, o):
        #debug("Visiting object of type %s." % type(o).__name__)
        
        # Get handler for the UFL class of o (type(o) may be an external subclass of the actual UFL class)
        h, visit_children_first = self._handlers[o._classid]
        
        #if not h:
        #    # Failed to find a handler! Should never happen, but will happen if a non-Expr object is visited.
        #    error("Can't handle objects of type %s" % str(type(o)))
        
        # Is this a handler that expects transformed children as input?
        if visit_children_first:
            # Yes, visit all children first and then call h.
            return h(o, *map(self.visit, o.operands()))
        
        # No, this is a handler that handles its own children
        # (arguments self and o, where self is already bound)
        return h(o)
    
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
    basis_function = wrap_terminal
    function = wrap_terminal
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

# Note:
# To avoid typing errors, the expressions for cofactor and deviatoric parts 
# below were created with the script tensoralgebrastrings.py under ufl/scripts/
class CompoundExpander(ReuseTransformer):
    "Expands compound expressions to equivalent representations using basic operators."
    def __init__(self, geometric_dimension):
        ReuseTransformer.__init__(self)
        self._dim = geometric_dimension
        if self._dim is None:
            warning("Got None for dimension, some compounds cannot be expanded.")

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
            return as_matrix([[-A[1,1],A[0,1]],[A[1,0],-A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([[-A[1,1]-A[2,2],A[0,1],A[0,2]],[A[1,0],-A[0,0]-A[2,2],A[1,2]],[A[2,0],A[2,1],-A[0,0]-A[1,1]]])
        error("dev(A) not implemented for dimension %s." % sh[0])
    
    def skew(self, o, A):
        i, j = indices(2)
        return as_matrix( (A[i,j] - A[j,i]) / 2, (i,j) )
    
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
        ii = Index()
        if a.rank() > 0:
            jj = tuple(indices(a.rank()))
            return as_tensor(a[jj].dx(ii), tuple((ii,)+jj))
        return as_tensor(a.dx(ii), (ii,))
    
    def curl(self, o, a):
        # o = curl a = "cross(nabla, a)"
        def c(i, j):
            return a[j].dx(i) - a[i].dx(j)
        return as_vector((c(1,2), c(2,0), c(0,1))) # FIXME: Verify this
    
    def rot(self, o, a):
        raise NotImplementedError # FIXME

class NotMultiLinearException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

class BasisFunctionDependencyExtracter(Transformer):
    def __init__(self):
        Transformer.__init__(self)
        self._empty = frozenset()
    
    def expr(self, o, *opdeps):
        "Default for nonterminals: nonlinear in all arguments."
        for d in opdeps:
            if d:
                raise NotMultiLinearException, repr(o)
        return self._empty
    
    def terminal(self, o):
        "Default for terminals: no dependency on basis functions."
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
    
    def basis_function(self, o):
        d = frozenset((o,))
        return frozenset((d,))
    
    def linear(self, o, a):
        "Nonterminals that are linear with a single argument."
        return a
    grad = linear
    div = linear
    curl = linear
    rot = linear
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
                # build frozenset cd with the combined BasisFunction dependencies from ad and bd
                cd = (ad | bd) - none
                # build frozenset cd with the combined BasisFunction dependencies from ad and bd
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

class IndexExpander(ReuseTransformer):
    """..."""
    def __init__(self):
        ReuseTransformer.__init__(self)
        self._components = Stack()
        self._index2value = StackDict()
    
    def component(self):
        "Return current component tuple."
        if self._components:
            return self._components.peek()
        return ()
    
    def terminal(self, x):
        if x.shape():
            return x[self.component()]
        return x
    
    def form_argument(self, x):
        if x.shape():
            # Get symmetry mapping if any
            e = x.element()
            s = None
            if isinstance(e, TensorElement):
                s = e.symmetry()
            if s is None:
                s = {}
            # Map component throught the symmetry mapping
            c = self.component()
            c = s.get(c, c)
            return x[c]
        return x
    
    def zero(self, x):
        # FIXME: These assertions may not always work out, figure out why!
        ufl_assert(len(x.shape()) == len(self.component()), "Component size mismatch.")
        #s = set(x.free_indices()) - set(self._index2value.keys())
        #ufl_assert(not s, "Free index set mismatch.")
        return x._uflclass()
    
    def scalar_value(self, x):
        # FIXME: These assertions may not always work out, figure out why!
        ufl_assert(len(x.shape()) == len(self.component()), "Component size mismatch.")
        #s = set(x.free_indices()) - set(self._index2value.keys())
        #ufl_assert(not s, "Free index set mismatch.")
        return x._uflclass(x.value())
    
    def index_sum(self, x):
        ops = []
        summand, multiindex = x.operands()
        index, = multiindex

        # TODO: For the list tensor purging algorithm, do something like:
        # if index not in self._to_expand:
        #     return self.expr(x, *[self.visit(o) for o in x.operands()])

        for value in range(x.dimension()):
            self._index2value.push(index, value)
            ops.append(self.visit(summand))
            self._index2value.pop()
        return sum(ops)
    
    def _multi_index(self, x):
        comp = []
        for i in x:
            if isinstance(i, FixedIndex):
                comp.append(i._value)
            elif isinstance(i, Index):
                comp.append(self._index2value[i])
        return tuple(comp)
    
    def multi_index(self, x):
        return x._uflclass(self._multi_index(x))
    
    def indexed(self, x):
        A, ii = x.operands()
        # Push new component built from index value map
        self._components.push(self._multi_index(ii))
        # Hide index values # TODO: This causes None to occur in _multi_index, need to make sure I've got this whole thing right...
        #for i in ii:
        #    if isinstance(i, Index):
        #        self._index2value.push(i, None)
        result = self.visit(A)
        # Un-hide index values
        #for i in ii:
        #    if isinstance(i, Index):
        #        self._index2value.pop()
        # Reset component
        self._components.pop()
        return result
    
    def component_tensor(self, x):
        # This function evaluates the tensor expression
        # with indices equal to the current component tuple
        expression, indices = x.operands()
        ufl_assert(expression.shape() == (), "Expecting scalar base expression.")
        
        # Update index map with component tuple values
        comp = self.component()
        ufl_assert(len(indices) == len(comp), "Index/component mismatch.")
        for i, v in izip(indices._indices, comp):
            self._index2value.push(i, v)
        self._components.push(())
        
        # Evaluate with these indices
        result = self.visit(expression)
        
        # Revert index map
        for _ in comp:
            self._index2value.pop()
        self._components.pop()
        return result
    
    def list_tensor(self, x):
        # Pick the right subtensor and subcomponent
        c = self.component()
        c0, c1 = c[0], c[1:]
        op = x.operands()[c0]
        # Evaluate subtensor with this subcomponent
        self._components.push(c1)
        r = self.visit(op)
        self._components.pop()
        return r

    def spatial_derivative(self, x):
        f, i = x.operands()
        ufl_assert(isinstance(f, (Terminal, x._uflclass)), "Expecting expand_derivatives to have been applied.")
        
        f = self.visit(f) # taking component if necessary
        
        j = self.visit(i) # mapping to constant if necessary
        
        result = self.reuse_if_possible(x, f, j)
        
        return result

# ------------ User interface functions

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
            error("No integrals left after transformation, cannot reconstruct form.") # TODO: Is this the right behaviour?
        return Form(newintegrals)
    elif isinstance(form, Expr):
        return transform(form)
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
            dim = cell.d
    return apply_transformer(e, CompoundExpander(dim))

def expand_indices(e):
    return apply_transformer(e, IndexExpander())

def purge_list_tensors(e):
    """Get rid of all ListTensor instances by expanding
    expressions to use their components directly.
    Will usually increase the size of the expression."""
    if has_type(e, ListTensor):
        return expand_indices(e) # FIXME: Only expand what's necessary to get rid of list tensors
    return e

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

def extract_basis_function_dependencies(e):
    "Extract a set of sets of basis_functions."
    ufl_assert(isinstance(e, Expr), "Expecting an Expr.")
    return BasisFunctionDependencyExtracter().visit(e)

