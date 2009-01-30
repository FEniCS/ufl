"""This module defines utilities for transforming
complete Forms into new related Forms."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-10-01 -- 2009-01-09"

# Modified by Anders Logg, 2008

from itertools import izip

from ufl.common import some_key, product
from ufl.log import error, warning
from ufl.assertions import ufl_assert

# All classes:
from ufl.basisfunction import BasisFunction
#from ufl.basisfunction import TestFunction, TrialFunction, BasisFunctions, TestFunctions, TrialFunctions
from ufl.scalar import IntValue
from ufl.function import Function, Constant
from ufl.form import Form
from ufl.variable import Variable
from ufl.integral import Integral

# Lists of all Expr classes
from ufl.classes import ufl_classes, terminal_classes, nonterminal_classes

# Other algorithms:
from ufl.algorithms.analysis import extract_basisfunctions, extract_coefficients
from ufl.algorithms.transformations import replace, Transformer, apply_transformer, transform_integrands

def compute_form_action(form, function):
    """Compute the action of a form on a Function.
    
    This works simply by replacing the last basisfunction
    with a Function on the same function space (element).
    The form returned will thus have one BasisFunction less 
    and one additional Function at the end.
    """
    bf = extract_basisfunctions(form)
    ufl_assert(len(bf) == 2, "Expecting bilinear form.")
    v, u = bf
    e = u.element()
    if function is None:
        function = Function(e)
    else:
        ufl_assert(function.element() == e, \
            "Trying to compute action of form on a "\
            "function in an incompatible element space.")
    return replace(form, {u:function})

class ArityAnalyser(Transformer): # TODO: Can we avoid the cache stuff? Seems a bit clumsy in retrospect.
    def __init__(self, arity):
        Transformer.__init__(self)
        self._arity_cache = {}
        self._arity = arity
        print "X\n"*100
    
    def expr(self, x, *ops):
        # FIXME: Other operators to implement particularly? Will see when this triggers...
        # The default is a nonlinear operator not accepting arity > 0
        for o in ops:
            arity = self._arity_cache[id(o)]
            ufl_assert(arity == 0, "Invalid arity %d for an operand of a %s." % (arity, x._uflclass))
        self._arity_cache[id(x)] = 0
        return x
    
    def terminal(self, x):
        # Default terminal behaviour doesn't modify
        self._arity_cache[id(x)] = 0
        return x
    
    def variable(self, x):
        e, l = x.operands()
        result = self._variable_cache.get(l)
        if result is None:
            e2 = self.visit(e)
            arity = self._arity_cache[id(e)]
            
            # Reuse or reconstruct variable
            if e is e2:
                result = x
            else:
                result = Variable(2, l)
            self._variable_cache[l] = result
            
            self._arity_cache[id(result)] = arity
        return result
    
    def basis_function(self, x):
        self._arity_cache[id(x)] = 1
        return x
    
    def linear_operator(self, x, arg):
        # A linear operator in a single argument accepting arity > 0, just passing it on
        arity = self._arity_cache[id(arg)]
        
        # Reuse or reconstruct
        if arg is x.operands()[0]:
            result = x
        else:
            result = x._uflclass(arg)
        
        self._arity_cache[id(result)] = arity
        return result
    
    # FIXME: List all linear operators (use subclassing to simplify stuff like this?)
    derivative = linear_operator
    transposed = linear_operator
    
    def product(self, x, *ops):
        # Sum the arities of all operands
        arity = 0
        for o in ops:
            arity += self._arity_cache[id(o)]
        
        # Reuse or reconstruct
        if all((a is b) for (a, b) in zip(ops, x.operands())):
            result = x
        else:
            result = x._uflclass(*ops)
        
        self._arity_cache[id(result)] = arity
        return result
    inner = product
    outer = product
    dot = product
    
    def index_sum(self, o, f, i):
        print "\n"*100
        FIXME
    
    def sum(self, x, *ops):
        # Split operands into separate lists based on form arity
        opgroups = {}
        for o in ops:
            arity = self._arity_cache[id(o)]
            if not arity in opgroups:
                opgroups[arity] = []
            opgroups[arity].append(o)

        # (TODO: The correctness of this needs to be verified/proven)
        
        # Delete operands with arity higher than we want
        k = opgroups.keys()
        for i in k:
            if i > self._arity:
                del opgroups[i]
        
        # If we got more than one arity left, pick the one we're after TODO: This can fail
        if len(opgroups) > 1:
            arity = self._arity
        else:
            arity = opgroups.keys()[0]
        ops = opgroups[arity]
        
        # Reuse or reconstruct
        ops2 = x.operands()
        if len(ops) == len(ops2) and all((a is b) for (a, b) in zip(ops, ops2)):
            result = x
        else:
            result = x._uflclass(*ops)
        
        self._arity_cache[id(result)] = arity
        return result

def compute_form_lhs(form):
    """Compute the left hand side of a form."""
    # TODO: Does this work? Test and finish!
    # TODO: Can we use extract_basisfunction_dependencies for this?
    #return apply_transformer(form, ArityAnalyser(2))
    r = 2
    aa = ArityAnalyser(r)
    def _transform(e):
        e = aa.visit(e)
        if r == aa._arity_cache[id(e)]:
            return e
        return IntValue(0)
    res = transform_integrands(form, _transform)
    return res

def compute_form_rhs(form):
    """Compute the right hand side of a form."""
    # TODO: Does this work? Test and finish!
    # TODO: Can we use extract_basisfunction_dependencies for this?
    #return apply_transformer(form, ArityAnalyser(1))
    r = 1
    aa = ArityAnalyser(r)
    def _transform(e):
        e = aa.visit(e)
        if r == aa._arity_cache[id(e)]:
            return e
        return IntValue(0)
    res = transform_integrands(form, _transform)
    return res

def compute_form_adjoint(form):
    """Compute the adjoint of a bilinear form.
    
    This works simply by swapping the first and last basisfunctions.
    """
    bf = extract_basisfunctions(form)
    ufl_assert(len(bf) == 2, "Expecting bilinear form.")
    v, u = bf
    return replace(form, {v:u, u:v})

#def compute_dirichlet_functional(form):
#    """Compute the Dirichlet functional of a form:
#    a(v,u;...) - L(v; ...) -> 0.5 a(v,v;...) - L(v;...)
#    
#    This assumes a bilinear form and works simply by
#    replacing the trial function with the test function.
#    The form returned will thus be a linear form.
#    """
#    warning("TODO: Don't know if this is correct or even useful, just picked up the name some place.")
#    return 0.5*compute_form_lhs(form) - compute_form_rhs(form)
#    #bf = extract_basisfunctions(form)
#    #ufl_assert(len(bf) == 2, "Expecting bilinear form.")
#    #v, u = bf
#    #return replace(form, {u:v})
