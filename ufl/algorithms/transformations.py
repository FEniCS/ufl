"""This module defines expression transformation utilities,
either converting UFL expressions to new UFL expressions or
converting UFL expressions to other representations."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2009-01-09"

from inspect import getargspec
from itertools import izip, chain

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import camel2underscore
from ufl.expr import Expr
from ufl.terminal import Terminal
from ufl.indexing import Index, indices, complete_shape
from ufl.tensors import as_tensor, as_matrix, as_vector
from ufl.variable import Variable
from ufl.form import Form
from ufl.integral import Integral
from ufl.classes import all_ufl_classes
from ufl.algorithms.analysis import extract_duplications

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
        error("Didn't find class %s among handlers." % c)
    return h(expression, *ops)

def transform_integrands(form, transformer):
    ufl_assert(isinstance(form, Form), "Expecting Form.")
    if isinstance(form, Form):
        newintegrals = []
        for integral in form.integrals():
            newintegrand = transformer(integral.integrand())
            newintegral= integral.reconstruct(integrand = newintegrand)
            newintegrals.append(newintegral)
        newform = Form(newintegrals)
        return newform

class Transformer(object):
    """Base class for a visitor-like algorithm design pattern used to 
    transform expression trees from one representation to another."""
    def __init__(self, variable_cache=None):
        self._variable_cache = {} if variable_cache is None else variable_cache
        self._handlers = {}
        
        # For all UFL classes
        for uc in all_ufl_classes:
            # Iterate over the inheritance chain (assumes that all UFL classes has an Expr subclass as the first superclass)
            for c in uc.mro():
                # Register class uc with handler for the first encountered superclass
                fname = camel2underscore(c.__name__)
                if hasattr(self, fname):
                    self.register(uc, getattr(self, fname))
                    break
    
    def register(self, classobject, function):
        self._handlers[classobject] = function
    
    def visit(self, o):
        # Get handler for the UFL class of o (type(o) may be an external subclass of the actual UFL class)
        h = self._handlers.get(o._uflid)
        if h:
            # Did we find a handler that expects transformed children as input?
            insp = getargspec(h)
            num_args = len(insp[0]) + int(insp[1] is not None)
            if num_args > 2:
                return h(o, *[self.visit(oo) for oo in o.operands()])
            # No, this is a handler that handles its own children (arguments self and o, where self is already bound).
            return h(o)
        # Failed to find a handler!
        raise RuntimeError("Can't handle objects of type %s" % str(type(o)))

    def reuse(self, o):
        "Always reuse Expr (ignore children)"
        return o
    
    def reuse_if_possible(self, o, *operands):
        "Reuse Expr if possible, otherwise recreate."
        return o if operands == o.operands() else o._uflid(*operands)
    
    def always_recreate(self, o, *operands):
        "Always recreate expr."
        return o._uflid(*operands)
    
    # Set default behaviour for terminals
    terminal = reuse

class ReuseTransformer(Transformer):
    def __init__(self, variable_cache=None):
        Transformer.__init__(self, variable_cache)
    
    # Set default Expr behaviour
    expr = Transformer.reuse_if_possible
    
    def variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        v = self._variable_cache.get(l)
        if v is None:
            # Visit the expression our variable represents
            e2 = self.visit(e)
            # Recreate Variable (with same label) only if necessary
            if e is e2:
                return o
            v = Variable(e2, l)
            self._variable_cache[l] = v
        return v

class CopyTransformer(Transformer):
    def __init__(self, variable_cache=None):
        Transformer.__init__(self, variable_cache)
    
    # Set default Expr behaviour
    expr = Transformer.always_recreate
    
    def variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.operands()
        v = self._variable_cache.get(l)
        if v is None:
            # Visit the expression our variable represents
            e2 = self.visit(e)
            # Always recreate Variable (with same label)
            v = Variable(e2, l)
            self._variable_cache[l] = v
        return v

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
        c = o._uflid
        operands = []
        for b in ops:
            if isinstance(b, c):
                operands.extend(b.operands())
            else:
                operands.append(b)
        return c(*operands)
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
                o = o._uflid(*ops)
            
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
    
    # ------------ Compound tensor operators
    
    def trace(self, o, A):
        i = Index()
        return A[i,i]
    
    def transposed(self, o, A):
        i, j = indices(2)
        return as_tensor(A[i, j], (j, i))
    
    def deviatoric(self, o, A):
        sh = complete_shape(A.shape(), self._dim)
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
            return a[i]*b[j]-a[j]*b[i]
        return as_vector((c(1,2), c(2,0), c(0,1)))
    
    def dot(self, o, a, b):
        i = Index()
        aa = a[i] if (a.rank() == 1) else a[...,i]
        bb = b[i] if (b.rank() == 1) else b[i,...]
        return aa*bb
    
    def inner(self, o, a, b):
        ufl_assert(a.rank() == b.rank())
        ii = indices(a.rank())
        return a[ii]*b[ii]
    
    def outer(self, o, a, b):
        ii = indices(a.rank())
        jj = indices(b.rank())
        return a[ii]*b[jj]
    
    def determinant(self, o, A):
        sh = complete_shape(A.shape(), self._dim)
        if len(sh) == 0:
            return A
        ufl_assert(sh[0] == sh[1], "Expecting square matrix.")
        def det2D(B, i, j, k, l):
            return B[i,k]*B[j,l]-B[i,l]*B[j,k]
        if sh[0] == 2:
            return det2D(A, 0, 1, 0, 1)
        if sh[0] == 3:
            # TODO: Verify this expression
            return A[0,0]*det2D(A, 1, 2, 1, 2) + \
                   A[0,1]*det2D(A, 1, 2, 2, 0) + \
                   A[0,2]*det2D(A, 1, 2, 0, 1)
        # TODO: Implement generally for all dimensions?
        error("Determinant not implemented for dimension %d." % self._dim)
    
    def cofactor(self, o, A):
        sh = complete_shape(A.shape(), self._dim)
        ufl_assert(sh[0] == sh[1], "Expecting square matrix.")
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
        raise NotImplementedError # TODO
    
    def rot(self, o, a):
        raise NotImplementedError # TODO

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
                e = x._uflid(*ops)
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

# ------------ User interface functions

def apply_transformer(e, transformer):
    if isinstance(e, Form):
        newintegrals = []
        for itg in e.integrals():
            newintegrand = transformer.visit(itg.integrand())
            newitg = itg.reconstruct(integrand = newintegrand)
            newintegrals.append(newitg)
        return Form(newintegrals)
    ufl_assert(isinstance(e, Expr), "Expecting Form or Expr.")
    return transformer.visit(e)

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

def flatten(e): # TODO: Fix or remove!
    """Convert an UFL expression to a new UFL expression, with sums 
    and products flattened from binary tree nodes to n-ary tree nodes."""
    warning("flatten doesn't work correctly for some indexed products, like (u[i]*v[i])*(q[i]*r[i])") 
    return apply_transformer(e, TreeFlattener())

def expand_compounds(e, dim=None):
    """Expand compound objects into basic operators.
    Requires e to have a well defined domain, 
    for the geometric dimension to be defined."""
    if dim is None:
        dim = e.cell().dim()
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

def purge_duplications(expression):
    """Replace any subexpressions in expression that
    occur more than once with a single instance."""
    return apply_transformer(expression, DuplicationPurger())

def extract_basisfunction_dependencies(e):
    "Extract a set of sets of basisfunctions."
    ufl_assert(isinstance(e, Expr), "Expecting an Expr.")
    return BasisFunctionDependencyExtracter().visit(e)

