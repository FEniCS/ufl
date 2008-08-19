"""This module defines automatic differentiation utilities."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-19-- 2008-08-19"

from collections import defaultdict

from ..output import ufl_assert, ufl_error

# All classes:
from ..base import UFLObject, Terminal, Number
from ..base import ZeroType, zero_tensor # Experimental!
from ..variable import Variable
from ..finiteelement import FiniteElementBase, FiniteElement, MixedElement, VectorElement, TensorElement
from ..basisfunctions import BasisFunction, Function, Constant
from ..geometry import FacetNormal
from ..indexing import MultiIndex, Indexed, Index
from ..tensors import ListVector, ListMatrix, Tensor
from ..algebra import Sum, Product, Division, Power, Mod, Abs
from ..tensoralgebra import Identity, Transposed, Outer, Inner, Dot, Cross, Trace, Determinant, Inverse, Deviatoric, Cofactor
from ..mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin
from ..restriction import Restricted, PositiveRestricted, NegativeRestricted
from ..differentiation import PartialDerivative, Diff, Grad, Div, Curl, Rot

# Lists of all UFLObject classes
#from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes
from ..classes import terminal_classes


def diff_handlers():
    """This function constructs a default handler dict for 
    compute_derivative. Nonterminal objects are reused if possible."""
    # Show a clear error message if we miss some types here:
    def not_implemented(x, *ops):
        ufl_error("No handler defined for %s in diff_handlers. Add to classes.py." % x.__class__)
    d = defaultdict(not_implemented)
    # Terminal objects are assumed independent of the differentiation
    # variable by default, and simply lifted to the pair (x, 0)
    def lift(x):
        return (x, zero_tensor(x.shape()))
    for c in terminal_classes:
        d[c] = lift
    # Differentiation rules for nonterminal objects should never need to be overridden:
    
    def diff_variable(x, *ops):
        return (x, FIXME) # HOW?
    d[Variable] = diff_variable
    
    def diff_indexed(x, *ops):
        return (x, FIXME) # LINEAR
    d[Indexed] = diff_indexed
    
    def diff_listvector(x, *ops):
        return (x, FIXME) # LINEAR
    d[ListVector] = diff_listvector

    def diff_listmatrix(x, *ops):
        return (x, FIXME) # LINEAR
    d[ListMatrix] = diff_listmatrix

    def diff_tensor(x, *ops):
        return (x, FIXME) # LINEAR
    d[Tensor] = diff_tensor

    def diff_sum(x, *ops):
        return (sum(o[0] for o in ops if not isinstance(o[0], ZeroType)),
                sum(o[1] for o in ops if not isinstance(o[1], ZeroType)))
    d[Sum] = diff_sum
    
    def diff_product(x, *ops):
        if any(isinstance(o[0], ZeroType) for o in ops):
            f = zero_tensor(x.shape())
        else:
            f = sum(o[0] for o in ops) # TODO: Reuse x if possible
        if any(isinstance(o[1], ZeroType) for o in ops):
            fp = zero_tensor(x.shape())
        else:
            fp = sum( product(o[0] for o in ops[:i]) * ops[i][1] * product(o[0] for o in ops[i+1:]) \
                      for i in range(len(ops)) )
        return (f, fp)
    d[Product] = diff_product

    def diff_division(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Division] = diff_division

    def diff_power(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Power] = diff_power

    def diff_mod(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Mod] = diff_mod

    def diff_abs(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Abs] = diff_abs

    def diff_transposed(x, *ops):
        return (x, FIXME) # LINEAR
    d[Transposed] = diff_transposed

    def diff_outer(x, *ops):
        return (x, FIXME) # COMPOUND
    d[Outer] = diff_outer

    def diff_inner(x, *ops):
        return (x, FIXME) # COMPOUND
    d[Inner] = diff_inner

    def diff_dot(x, *ops):
        return (x, FIXME) # COMPOUND
    d[Dot] = diff_dot

    def diff_cross(x, *ops):
        return (x, FIXME) # COMPOUND
    d[Cross] = diff_cross

    def diff_trace(x, *ops):
        return (x, FIXME) # LINEAR
    d[Trace] = diff_trace

    def diff_determinant(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Determinant] = diff_determinant

    def diff_inverse(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Inverse] = diff_inverse

    def diff_deviatoric(x, *ops):
        return (x, FIXME) # LINEAR?
    d[Deviatoric] = diff_deviatoric

    def diff_cofactor(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Cofactor] = diff_cofactor

    def diff_sqrt(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Sqrt] = diff_sqrt
    
    def diff_exp(x, *ops):
        return (x, ops[1]*exp(ops[0])) # NONLINEAR
    d[Exp] = diff_exp
    
    def diff_ln(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Ln] = diff_ln
    
    def diff_cos(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Cos] = diff_cos
    
    def diff_sin(x, *ops):
        return (x, FIXME) # NONLINEAR
    d[Sin] = diff_sin

    def diff_positiverestricted(x, *ops):
        return (x, FIXME) # WHAT?
    d[PositiveRestricted] = diff_positiverestricted

    def diff_negativerestricted(x, *ops):
        return (x, FIXME) # WHAT?
    d[NegativeRestricted] = diff_negativerestricted

    def diff_partialderivative(x, *ops):
        return (x, FIXME) # LINEAR
    d[PartialDerivative] = diff_partialderivative

    def diff_diff(x, *ops):
        return (x, FIXME) # LINEAR
    d[Diff] = diff_diff

    def diff_grad(x, *ops):
        return (x, FIXME) # LINEAR
    d[Grad] = diff_grad

    def diff_div(x, *ops):
        return (x, FIXME) # LINEAR
    d[Div] = diff_div

    def diff_curl(x, *ops):
        return (x, FIXME) # LINEAR
    d[Curl] = diff_curl

    def diff_rot(x, *ops):
        return (x, FIXME) # LINEAR
    d[Rot] = diff_rot

    return d

def _compute_derivative(expression, handlers):
    "Returns a tuple (expression, expression_prime)."
    if isinstance(expression, Terminal):
        ops = ()
    else:
        ops = [_compute_derivative(o, handlers) for o in expression.operands()]
    return handlers[expression.__class__](expression, *ops)


def compute_derivative(expression, w):
    "Returns a tuple (expression, expression_prime)."
    ufl_assert(isinstance(w, Function), "Expecting Function to differentiate w.r.t.")
    wprime = BasisFunction(w.element())
    handlers = diff_handlers()
    def diff_function(x):
        if x == w:
            return (w, wprime)
        else:
            return zero_tensor(x.shape())
    return _compute_derivative(expression, handlers)

