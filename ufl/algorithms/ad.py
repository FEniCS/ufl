"""This module defines automatic differentiation utilities."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-08-19-- 2008-08-20"

from collections import defaultdict

from ..output import ufl_assert, ufl_error

# All classes:
from ..base import UFLObject, Terminal, FloatValue
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
from ..conditional import EQ, NE, LE, GE, LT, GT, Conditional # FIXME: Handle these

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
    
    # This should work for all operators that commute with d/dw:
    def diff_commute(x, *ops):
        ufl_assert(len(ops) == 1, "Logic breach in diff_commute, len(ops) = %d." % len(ops))
        return (x, x.__class__(ops[0][1]))
    
    # TODO: Can we use this anywhere? A bit dangerous.
    #def diff_commute_multiple_arguments(x, *ops):
    #    return (x, x.__class__(*[o[1] for o in ops]))
    
    # These differentiation rules for nonterminal objects should probably never need to be overridden:
    #def diff_variable(x, *ops):
    #    return (x, FIXME) # HOW?
    #d[Variable] = diff_variable
    d[Variable] = diff_commute # FIXME: Is this ok for variable? What about caching and reuse?

    def diff_multi_index(x, *ops):
        return (x, None) # x' here should never be used
    d[MultiIndex] = diff_multi_index

    def diff_indexed(x, *ops):
        A, i = ops
        return (x, A[1][i[0]])
    d[Indexed] = diff_indexed
    
    def diff_listvector(x, *ops):
        return (x, ListVector(*[o[1] for o in ops]))
    d[ListVector] = diff_listvector 

    def diff_matrix(x, *ops):
        return (x, ListMatrix(*[o[1] for o in ops]))
    d[ListMatrix] = diff_listmatrix

    def diff_tensor(x, *ops):
        A, i = ops
        if isinstance(A[1], ZeroType):
            return (x, zero_tensor(x.shape()))
        return (x, Tensor(A[1], i[0]) )
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
        f, fp = ops[0]
        g, gp = ops[1]
        return (x, (fp*g-f*gp)/g**2)
    d[Division] = diff_division

    def diff_power(x, *ops):
        f, fp = ops[0]
        g, gp = ops[1]
        ufl_assert(f.rank() == 0 and g.rank() == 0, "Expecting scalar expressions f,g in f**g.")
        # x = f**g
        f_const = isinstance(fp, ZeroType)
        g_const = isinstance(gp, ZeroType)
        # Case: x = const ** const = const
        if f_const and g_const:
            return (x, zero())
        # Case: x = f(x) ** const
        if g_const:
            # x' = g f'(x) f(x)**(g-1)
            if isinstance(g, ZeroType) or isinstance(f, ZeroType) or f_const:
                return (x, zero())
            return (x, g*fp*f**(g-1.0))
        # Case: x = f ** g(x)
        if isinstance(fp, ZeroType):
            return (x, gp*ln(f)*x)
        ufl_error("diff_power not implemented for case d/dx [ f(x)**g(x) ].")
        return (x, FIXME)
    d[Power] = diff_power

    def diff_mod(x, *ops):
        ufl_error("diff_mod not implemented")
        return (x, FIXME)
    d[Mod] = diff_mod

    def diff_abs(x, *ops):
        ufl_error("diff_abs not implemented")
        return (x, FIXME)
    d[Abs] = diff_abs
    
    d[Transposed] = diff_commute
    
    def diff_outer(x, *ops):
        ufl_error("diff_outer not implemented")
        return (x, FIXME) # COMPOUND
    d[Outer] = diff_outer

    def diff_inner(x, *ops):
        ufl_error("diff_inner not implemented")
        return (x, FIXME) # COMPOUND
    d[Inner] = diff_inner

    def diff_dot(x, *ops):
        ufl_error("diff_dot not implemented")
        return (x, FIXME) # COMPOUND
    d[Dot] = diff_dot

    def diff_cross(x, *ops):
        u, up = ops[0]
        v, vp = ops[1]
        ufl_error("diff_cross not implemented")
        return (x, FIXME) # COMPOUND
    d[Cross] = diff_cross

    d[Trace] = diff_commute

    def diff_determinant(x, *ops):
        A, Ap = ops[0]
        ufl_error("diff_determinant not implemented")
        return (x, FIXME)
    d[Determinant] = diff_determinant

    # Derivation:
    # 0 = d/dx [Ainv*A] = Ainv' * A + Ainv * A'
    # Ainv' * A = - Ainv * A'
    # Ainv' = - Ainv * A' * Ainv
    def diff_inverse(x, *ops):
        A, Ap = ops[0]
        return (x, -x*Ap*x)
    d[Inverse] = diff_inverse

    d[Deviatoric] = diff_commute

    def diff_cofactor(x, *ops):
        A, Ap = ops[0]
        ufl_error("diff_cofactor not implemented")
        return (x, FIXME) # NONLINEAR
    d[Cofactor] = diff_cofactor

    # Mathfunctions:
    def diff_sqrt(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, ZeroType): return (x, zero())
        return (x, 0.5*fp/sqrt(f))
    d[Sqrt] = diff_sqrt
    
    def diff_exp(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, ZeroType): return (x, zero())
        return (x, fp*exp(f))
    d[Exp] = diff_exp
    
    def diff_ln(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, ZeroType): return (x, zero())
        ufl_assert(not isinstance(f, ZeroType), "Division by zero.")
        return (x, fp/f)
    d[Ln] = diff_ln
    
    def diff_cos(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, ZeroType): return (x, zero())
        return (x, -fp*sin(f))
    d[Cos] = diff_cos
    
    def diff_sin(x, *ops):
        f, fp = ops[0]
        if isinstance(fp, ZeroType): return (x, zero())
        return (x, fp*cos(f))
    d[Sin] = diff_sin

    # Restrictions
    def diff_positiverestricted(x, *ops):
        return (x, FIXME) # TODO: What is d(v+)/dw ?
    d[PositiveRestricted] = diff_positiverestricted

    def diff_negativerestricted(x, *ops):
        return (x, FIXME) # TODO: What is d(v+)/dw ?
    d[NegativeRestricted] = diff_negativerestricted
    
    # Derivatives
    d[PartialDerivative] = diff_commute
    d[Diff] = diff_commute
    d[Grad] = diff_commute
    d[Div] = diff_commute
    d[Curl] = diff_commute
    d[Rot] = diff_commute

    # Conditionals
    def diff_condition(x, *ops):
        return (x, 0)
    d[EQ] = diff_condition
    d[NE] = diff_condition
    d[LE] = diff_condition
    d[GE] = diff_condition
    d[LT] = diff_condition
    d[GT] = diff_condition
    def diff_conditional(x, *ops):
        c, l, r = ops
        if not isinstance(c[1], ZeroType):
            ufl_warning("Differentiating a conditional with a condition that is not constant w.r.t. the differentiation variable.")
        if isinstance(l[1], ZeroType) and isinstance(r[1], ZeroType):
            return (x, zero_tensor(x.shape()))
        return (x, conditional(c[0], l[1], r[1]))
    d[Conditional] = diff_conditional

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


# FIXME: Need algorithm to apply AD to all kinds of derivatives! In particular, PartialDerivative and Diff.

