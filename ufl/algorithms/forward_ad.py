# -*- coding: utf-8 -*-
"""Forward mode AD implementation."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009.
# Modified by Garth N. Wells, 2010.
# Modified by Kristian B. Oelgaard, 2011
# Modified by Jan Blechta, 2012.

from six.moves import xrange as range
from six.moves import zip

from math import pi
from ufl.log import error, warning, debug
from ufl.assertions import ufl_assert
from ufl.utils.sequences import unzip
from ufl.utils.dicts import subdict
from ufl.utils.formatting import lstr

# All classes:
from ufl.core.terminal import Terminal
from ufl.constantvalue import ConstantValue, Zero, IntValue, Identity,\
    is_true_ufl_scalar, is_ufl_scalar
from ufl.variable import Variable
from ufl.coefficient import Coefficient, FormArgument
from ufl.core.multiindex import MultiIndex, Index, FixedIndex, indices
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.tensors import ListTensor, ComponentTensor, as_tensor, as_scalar, unit_indexed_tensor, unwrap_list_tensor
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Transposed, Outer, Inner, Dot, Cross, Trace, \
    Determinant, Inverse, Deviatoric, Cofactor
from ufl.mathfunctions import MathFunction, Sqrt, Exp, Ln, Cos, Sin, Tan, Acos, Asin, Atan, Atan2, Erf, BesselJ, BesselY, BesselI, BesselK
from ufl.restriction import Restricted, PositiveRestricted, NegativeRestricted
from ufl.differentiation import Derivative, CoefficientDerivative,\
    VariableDerivative, Grad
from ufl.conditional import EQ, NE, LE, GE, LT, GT, Conditional
from ufl.exprcontainers import ExprList, ExprMapping
from ufl.operators import dot, inner, outer, lt, eq, conditional, sign, \
    sqrt, exp, ln, cos, sin, tan, cosh, sinh, tanh, acos, asin, atan, atan_2, \
    erf, bessel_J, bessel_Y, bessel_I, bessel_K, \
    cell_avg, facet_avg
from ufl.algorithms.transformer import Transformer
from ufl.domain import find_geometric_dimension
from ufl.checks import is_cellwise_constant


class ForwardAD(Transformer):
    def __init__(self, var_shape, cache=None):
        Transformer.__init__(self)
        self._var_shape = var_shape
        self._variable_cache = {} if cache is None else cache

    def _debug_visit(self, o):
        "Debugging hook, enable this by renaming to 'visit'."
        r = Transformer.visit(self, o)
        f, df = r
        if f is not o:
            debug("In ForwardAD.visit, didn't get back o:")
            debug("  o:  %s" % str(o))
            debug("  f:  %s" % str(f))
            debug("  df: %s" % str(df))
        fi_diff = set(f.ufl_free_indices) ^ set(df.ufl_free_indices)
        if fi_diff:
            debug("In ForwardAD.visit, got free indices diff:")
            debug("  o:  %s" % str(o))
            debug("  f:  %s" % str(f))
            debug("  df: %s" % str(df))
            debug("  f.fi():  %s" % lstr(f.ufl_free_indices))
            debug("  df.fi(): %s" % lstr(df.ufl_free_indices))
            debug("  fi_diff: %s" % str(fi_diff))
        return r

    def _make_zero_diff(self, o):
        # Define a zero with the right shape and indices
        return Zero(o.ufl_shape + self._var_shape, o.ufl_free_indices, o.ufl_index_dimensions)

    # --- Default rules

    def expr(self, o):
        error("Missing ForwardAD handler for type %s" % str(type(o)))

    def terminal(self, o):
        """Terminal objects are assumed independent of the differentiation
        variable by default, and simply 'lifted' to the pair (o, 0).
        Depending on the context, override this with custom rules for
        non-zero derivatives."""
        fp = self._make_zero_diff(o)
        return (o, fp)

    def variable(self, o):
        """Variable objects are just 'labels', so by default the derivative
        of a variable is the derivative of its referenced expression."""
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.ufl_operands
        r = self._variable_cache.get(l) # cache contains (v, vp) tuple
        if r is not None:
            return r

        # Visit the expression our variable represents
        e2, vp = self.visit(e)

        # Recreate Variable (with same label) only if necessary
        v = self.reuse_if_possible(o, e2, l)

        # Cache and return (v, vp) tuple
        r = (v, vp)
        self._variable_cache[l] = r
        return r

    # --- Indexing and component handling

    def multi_index(self, o):
        return (o, None) # oprime here should never be used, this might even not be called?

    def indexed(self, o):
        A, jj = o.ufl_operands
        A2, Ap = self.visit(A)
        o = self.reuse_if_possible(o, A2, jj)

        if isinstance(Ap, Zero):
            op = self._make_zero_diff(o)
        else:
            r = len(Ap.ufl_shape) - len(jj)
            if r:
                ii = indices(r)
                op = Indexed(Ap, MultiIndex(jj.indices() + ii))
                op = as_tensor(op, ii)
            else:
                op = Indexed(Ap, jj)
        return (o, op)

    def list_tensor(self, o, *ops):
        ops, dops = unzip(ops)
        o = self.reuse_if_possible(o, *ops)
        op = ListTensor(*dops)
        return (o, op)

    def component_tensor(self, o):
        A, ii = o.ufl_operands
        A, Ap = self.visit(A)
        o = self.reuse_if_possible(o, A, ii)

        if isinstance(Ap, Zero):
            op = self._make_zero_diff(o)
        else:
            Ap, jj = as_scalar(Ap)
            op = as_tensor(Ap, ii.indices() + jj)
        return (o, op)

    # --- Algebra operators

    def index_sum(self, o):
        A, i = o.ufl_operands
        A2, Ap = self.visit(A)
        o = self.reuse_if_possible(o, A2, i)
        op = IndexSum(Ap, i)
        return (o, op)

    def sum(self, o, *ops):
        ops, opsp = unzip(ops)
        o2 = self.reuse_if_possible(o, *ops)
        op = sum(opsp[1:], opsp[0])
        return (o2, op)

    def product(self, o, *ops):
        # Start with a zero with the right shape and indices
        fp = self._make_zero_diff(o)
        # Get operands and their derivatives
        ops2, dops2 = unzip(ops)
        o = self.reuse_if_possible(o, *ops2)
        for i in range(len(ops)):
            # Get scalar representation of differentiated value of operand i
            dop = dops2[i]
            dop, ii = as_scalar(dop)
            # Replace operand i with its differentiated value in product
            fpoperands = ops2[:i] + [dop] + ops2[i+1:]
            p = Product(*fpoperands)
            # Wrap product in tensor again
            if ii:
                p = as_tensor(p, ii)
            # Accumulate terms
            fp += p
        return (o, fp)

    def division(self, o, a, b):
        f, fp = a
        g, gp = b
        o = self.reuse_if_possible(o, f, g)

        ufl_assert(is_ufl_scalar(f), "Not expecting nonscalar nominator")
        ufl_assert(is_true_ufl_scalar(g), "Not expecting nonscalar denominator")

        #do_df = 1/g
        #do_dg = -h/g
        #op = do_df*fp + do_df*gp
        #op = (fp - o*gp) / g

        # Get o and gp as scalars, multiply, then wrap as a tensor again
        so, oi = as_scalar(o)
        sgp, gi = as_scalar(gp)
        o_gp = so*sgp
        if oi or gi:
            o_gp = as_tensor(o_gp, oi + gi)
        op = (fp - o_gp) / g

        return (o, op)

    def power(self, o, a, b):
        f, fp = a
        g, gp = b

        # Debugging prints, should never happen:
        if not is_true_ufl_scalar(f):
            print(":"*80)
            print("f =", str(f))
            print("g =", str(g))
            print(":"*80)
        ufl_assert(is_true_ufl_scalar(f), "Expecting scalar expression f in f**g.")
        ufl_assert(is_true_ufl_scalar(g), "Expecting scalar expression g in f**g.")

        # Derivation of the general case: o = f(x)**g(x)
        #
        #do_df = g * f**(g-1)
        #do_dg = ln(f) * f**g
        #op = do_df*fp + do_dg*gp
        #
        #do_df = o * g / f # f**g * g / f
        #do_dg = ln(f) * o
        #op = do_df*fp + do_dg*gp

        # Got two possible alternatives here:
        if True: # This version looks better.
            # Rewriting o as f*f**(g-1) we can do:
            f_g_m1 = f**(g-1)
            op = f_g_m1*(fp*g + f*ln(f)*gp)
            # In this case we can rewrite o using new subexpression
            o = f*f_g_m1
        else:
            # Pulling o out gives:
            op = o*(fp*g/f + ln(f)*gp)
            # This produces expressions like (1/w)*w**5 instead of w**4
            # If we do this, we reuse o
            o = self.reuse_if_possible(o, f, g)

        return (o, op)

    def abs(self, o, a):
        f, fprime = a
        o = self.reuse_if_possible(o, f)
        #oprime = conditional(eq(f, 0), 0, Product(sign(f), fprime))
        oprime = sign(f)*fprime
        return (o, oprime)

    # --- Mathfunctions

    def math_function(self, o, a):
        if hasattr(o, 'derivative'): # FIXME: Introduce a UserOperator type instead of this hack
            f, fp = a
            o = self.reuse_if_possible(o, f)
            op = fp * o.derivative()
            return (o, op)
        error("Unknown math function.")

    def sqrt(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp / (2*o)
        return (o, op)

    def exp(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*o
        return (o, op)

    def ln(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        ufl_assert(not isinstance(f, Zero), "Division by zero.")
        return (o, fp/f)

    def cos(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = -fp*sin(f)
        return (o, op)

    def sin(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*cos(f)
        return (o, op)

    def tan(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*2.0/(cos(2.0*f) + 1.0)
        return (o, op)

    def cosh(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*sinh(f)
        return (o, op)

    def sinh(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*cosh(f)
        return (o, op)

    def tanh(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        def sech(y):
            return (2.0*cosh(y)) / (cosh(2.0*y) + 1.0)
        op = fp*sech(f)**2
        return (o, op)

    def acos(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = -fp/sqrt(1.0 - f**2)
        return (o, op)

    def asin(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp/sqrt(1.0 - f**2)
        return (o, op)

    def atan(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp/(1.0 + f**2)
        return (o, op)

    def atan_2(self, o, a, b):
        f, fp = a
        g, gp = b
        o = self.reuse_if_possible(o, f, g)
        op = (g*fp-f*gp)/(f**2+g**2)
        return (o, op)

    def erf(self, o, a):
        f, fp = a
        o = self.reuse_if_possible(o, f)
        op = fp*(2.0/sqrt(pi)*exp(-f**2))
        return (o, op)

    def bessel_j(self, o, nu, x):
        nu, dummy = nu
        if not (dummy is None or isinstance(dummy, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")
        f, fp = x
        o = self.reuse_if_possible(o, nu, f)
        if isinstance(nu, Zero):
            op = -bessel_J(1, f)
        else:
            op = 0.5 * (bessel_J(nu-1, f) - bessel_J(nu+1, f))
        return (o, op*fp)

    def bessel_y(self, o, nu, x):
        nu, dummy = nu
        if not (dummy is None or isinstance(dummy, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")
        f, fp = x
        o = self.reuse_if_possible(o, nu, f)
        if isinstance(nu, Zero):
            op = -bessel_Y(1, f)
        else:
            op = 0.5 * (bessel_Y(nu-1, f) - bessel_Y(nu+1, f))
        return (o, op*fp)

    def bessel_i(self, o, nu, x):
        nu, dummy = nu
        if not (dummy is None or isinstance(dummy, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")
        f, fp = x
        o = self.reuse_if_possible(o, nu, f)
        if isinstance(nu, Zero):
            op = bessel_I(1, f)
        else:
            op = 0.5 * (bessel_I(nu-1, f) + bessel_I(nu+1, f))
        return (o, op*fp)

    def bessel_k(self, o, nu, x):
        nu, dummy = nu
        if not (dummy is None or isinstance(dummy, Zero)):
            error("Differentiation of bessel function w.r.t. nu is not supported.")
        f, fp = x
        o = self.reuse_if_possible(o, nu, f)
        if isinstance(nu, Zero):
            op = -bessel_K(1, f)
        else:
            op = -0.5 * (bessel_K(nu-1, f) + bessel_K(nu+1, f))
        return (o, op*fp)

    # --- Restrictions

    def restricted(self, o, a):
        # Restriction and differentiation commutes.
        f, fp = a
        o = self.reuse_if_possible(o, f)
        if isinstance(fp, ConstantValue):
            return (o, fp) # TODO: Necessary? Can't restriction simplify directly instead?
        else:
            return (o, fp(o._side)) # (f+-)' == (f')+-

    def cell_avg(self, o, a):
        # Cell average of a single function and differentiation commutes.
        f, fp = a
        o = self.reuse_if_possible(o, f)
        if isinstance(fp, ConstantValue):
            return (o, fp) # TODO: Necessary? Can't CellAvg simplify directly instead?
        else:
            return (o, cell_avg(fp))

    def facet_avg(self, o, a):
        # Facet average of a single function and differentiation commutes.
        f, fp = a
        o = self.reuse_if_possible(o, f)
        if isinstance(fp, ConstantValue):
            return (o, fp) # TODO: Necessary? Can't FacetAvg simplify directly instead?
        else:
            return (o, facet_avg(fp))

    # --- Conditionals

    def binary_condition(self, o, l, r):
        o = self.reuse_if_possible(o, l[0], r[0])
        #if any(not (isinstance(op[1], Zero) or op[1] is None) for op in (l, r)):
        #    warning("Differentiating a conditional with a condition "\
        #                "that depends on the differentiation variable."\
        #                "Assuming continuity of conditional. The condition "\
        #                "will not be differentiated.")
        oprime = None # Shouldn't be used anywhere
        return (o, oprime)

    def not_condition(self, o, c):
        o = self.reuse_if_possible(o, c[0])
        #if not (isinstance(c[1], Zero) or c[1] is None):
        #    warning("Differentiating a conditional with a condition "\
        #                "that depends on the differentiation variable."\
        #                "Assuming continuity of conditional. The condition "\
        #                "will not be differentiated.")
        oprime = None # Shouldn't be used anywhere
        return (o, oprime)

    def conditional(self, o, c, t, f):
        o = self.reuse_if_possible(o, c[0], t[0], f[0])
        if isinstance(t[1], Zero) and isinstance(f[1], Zero):
            # Assuming t[1] and f[1] have the same indices here, which should be the case
            op = t[1]
        else:
            # Placing t[1],f[1] outside here to avoid getting arguments inside conditionals
            op = conditional(c[0], 1, 0)*t[1] + conditional(c[0], 0, 1)*f[1]
        return (o, op)

    def max_value(self, o, x, y):
        #d/dx max(f, g) =
        # f > g: df/dx
        # f < g: dg/dx
        op = conditional(x[0] > y[0], x[1], y[1])
        return (o, op)

    def min_value(self, o, x, y):
        #d/dx min(f, g) =
        # f < g: df/dx
        # else: dg/dx
        op = conditional(x[0] < y[0], x[1], y[1])
        return (o, op)

    # --- Other derivatives

    def derivative(self, o):
        error("This should never occur.")

    def grad(self, o):
        error("FIXME")

# TODO: Add a ReferenceGradAD ruleset
class GradAD(ForwardAD):
    def __init__(self, geometric_dimension, cache=None):
        ForwardAD.__init__(self, var_shape=(geometric_dimension,), cache=cache)

    def geometric_quantity(self, o):
        "Represent grad(g) as Grad(g)."
        # Collapse gradient of cellwise constant function to zero
        if is_cellwise_constant(o):
            return self.terminal(o)
        return (o, Grad(o))

    def spatial_coordinate(self, o):
        "Gradient of x w.r.t. x is Id."
        if not hasattr(self, '_Id'):
            gdim = find_geometric_dimension(o)
            self._Id = Identity(gdim)
        return (o, self._Id)

    def cell_coordinate(self, o):
        "Gradient of X w.r.t. x is K. But I'm not sure if we want to allow this."
        error("This has not been validated. Does it make sense to do this here?")
        K = JacobianInverse(o.ufl_domain())
        return (o, K)

    # TODO: Implement rules for some of these types?

    def facet_coordinate(self, o):
        error("Not expecting this low level type in AD.")

    def jacobian(self, o):
        error("Not expecting this low level type in AD.")

    def jacobian_determinant(self, o):
        error("Not expecting this low level type in AD.")

    def jacobian_inverse(self, o):
        error("Not expecting this low level type in AD.")

    def facet_jacobian(self, o):
        error("Not expecting this low level type in AD.")

    def facet_jacobian_determinant(self, o):
        error("Not expecting this low level type in AD.")

    def facet_jacobian_inverse(self, o):
        error("Not expecting this low level type in AD.")

    def argument(self, o):
        "Represent grad(f) as Grad(f)."
        # Collapse gradient of cellwise constant function to zero
        # FIXME: Enable this after fixing issue#13
        #if is_cellwise_constant(o):
        #    return zero(...) # TODO: zero annotated with argument
        return (o, Grad(o))

    def coefficient(self, o):
        "Represent grad(f) as Grad(f)."
        # Collapse gradient of cellwise constant function to zero
        if is_cellwise_constant(o):
            return self.terminal(o)
        return (o, Grad(o))

    def grad(self, o):
        "Represent grad(grad(f)) as Grad(Grad(f))."

        # TODO: Not sure how to detect that gradient of f is cellwise constant.
        #       Can we trust element degrees?
        #if is_cellwise_constant(o):
        #    return self.terminal(o)
        # TODO: Maybe we can ask "f.has_derivatives_of_order(n)" to check
        #       if we should make a zero here?
        # 1) n = count number of Grads, get f
        # 2) if not f.has_derivatives(n): return zero(...)

        f, = o.ufl_operands
        ufl_assert(isinstance(f, (Grad, Terminal)),
                   "Expecting derivatives of child to be already expanded.")
        return (o, Grad(o))

class VariableAD(ForwardAD):
    def __init__(self, var, cache=None):
        ForwardAD.__init__(self, var_shape=var.ufl_shape, cache=cache)
        ufl_assert(not var.ufl_free_indices, "Differentiation variable cannot have free indices.")
        self._variable = var

    def grad(self, o):
        # If we hit this type, it has already been propagated
        # to a coefficient, so it cannot depend on the variable. # FIXME: Assert this!
        return self.terminal(o)

    def _make_self_diff_identity(self, o):
        sh = o.ufl_shape
        res = None
        if sh == ():
            # Scalar dv/dv is scalar
            return IntValue(1)
        elif len(sh) == 1:
            # Vector v makes dv/dv the identity matrix
            return Identity(sh[0])
        else:
            # Tensor v makes dv/dv some kind of higher rank identity tensor
            ind1 = ()
            ind2 = ()
            for d in sh:
                i, j = indices(2)
                dij = Identity(d)[i, j]
                if res is None:
                    res = dij
                else:
                    res *= dij
                ind1 += (i,)
                ind2 += (j,)
            fp = as_tensor(res, ind1 + ind2)

        return fp

    def variable(self, o):
        # Check cache
        e, l = o.ufl_operands
        c = self._variable_cache.get(l)

        if c is not None:
            return c

        if o.label() == self._variable.label():
            # dv/dv = identity of rank 2*rank(v)
            op = self._make_self_diff_identity(o)
        else:
            # differentiate expression behind variable
            e2, ep = self.visit(e)
            op = ep
            if not e2 == e:
                o = Variable(e2, l)
        # return variable and derivative of its expression
        c = (o, op)
        self._variable_cache[l] = c
        return c

class CoefficientAD(ForwardAD):
    "Apply AFD (Automatic Functional Differentiation) to expression."
    def __init__(self, coefficients, arguments, coefficient_derivatives, cache=None):
        ForwardAD.__init__(self, var_shape=(), cache=cache)
        ufl_assert(isinstance(coefficients, ExprList), "Expecting a ExprList.")
        ufl_assert(isinstance(arguments, ExprList), "Expecting a ExprList.")
        ufl_assert(isinstance(coefficient_derivatives, ExprMapping), "Expecting a ExprList.")
        self._v = arguments
        self._w = coefficients
        cd = coefficient_derivatives.ufl_operands
        self._cd = dict((cd[2*i], cd[2*i+1]) for i in range(len(cd)//2))

    def coefficient(self, o):
        # Define dw/dw := d/ds [w + s v] = v

        debug("In CoefficientAD.coefficient:")
        debug("o = %s" % o)
        debug("self._w = %s" % self._w)
        debug("self._v = %s" % self._v)

        # Find o among w
        for (w, v) in zip(self._w, self._v):
            if o == w:
                return (w, v)

        # If o is not among coefficient derivatives, return do/dw=0
        oprimesum = Zero(o.ufl_shape)
        oprimes = self._cd.get(o)
        if oprimes is None:
            if self._cd:
                # TODO: Make it possible to silence this message in particular?
                #       It may be good to have for debugging...
                warning("Assuming d{%s}/d{%s} = 0." % (o, self._w))
        else:
            # Make sure we have a tuple to match the self._v tuple
            if not isinstance(oprimes, tuple):
                oprimes = (oprimes,)
                ufl_assert(len(oprimes) == len(self._v), "Got a tuple of arguments, "+\
                               "expecting a matching tuple of coefficient derivatives.")

            # Compute do/dw_j = do/dw_h : v.
            # Since we may actually have a tuple of oprimes and vs in a
            # 'mixed' space, sum over them all to get the complete inner
            # product. Using indices to define a non-compound inner product.
            for (oprime, v) in zip(oprimes, self._v):
                so, oi = as_scalar(oprime)
                rv = len(v.ufl_shape)
                oi1 = oi[:-rv]
                oi2 = oi[-rv:]
                prod = so*v[oi2]
                if oi1:
                    oprimesum += as_tensor(prod, oi1)
                else:
                    oprimesum += prod

        # Example:
        # (f : g) -> (dfdu : v) : g + ditto
        # shape(f) == shape(g) == shape(dfdu : v)
        # shape(dfdu) == shape(f) + shape(v)

        return (o, oprimesum)

    def grad(self, g):
        # If we hit this type, it has already been propagated
        # to a coefficient (or grad of a coefficient), # FIXME: Assert this!
        # so we need to take the gradient of the variation or return zero.
        # Complications occur when dealing with derivatives w.r.t. single components...

        # Figure out how many gradients are around the inner terminal
        ngrads = 0
        o = g
        while isinstance(o, Grad):
            o, = o.ufl_operands
            ngrads += 1
        if not isinstance(o, FormArgument):
            error("Expecting gradient of a FormArgument, not %r" % (o,))

        def apply_grads(f):
            if not isinstance(f, FormArgument):
                print(','*60)
                print(f)
                print(o)
                print(g)
                print(','*60)
                error("What?")
            for i in range(ngrads):
                f = Grad(f)
            return f

        # Find o among all w without any indexing, which makes this easy
        for (w, v) in zip(self._w, self._v):
            if o == w and isinstance(v, FormArgument):
                # Case: d/dt [w + t v]
                return (g, apply_grads(v))

        # If o is not among coefficient derivatives, return do/dw=0
        gprimesum = Zero(g.ufl_shape)

        def analyse_variation_argument(v):
            # Analyse variation argument
            if isinstance(v, FormArgument):
                # Case: d/dt [w[...] + t v]
                vval, vcomp = v, ()
            elif isinstance(v, Indexed):
                # Case: d/dt [w + t v[...]]
                # Case: d/dt [w[...] + t v[...]]
                vval, vcomp = v.ufl_operands
                vcomp = tuple(vcomp)
            else:
                error("Expecting argument or component of argument.")
            ufl_assert(all(isinstance(k, FixedIndex) for k in vcomp),
                       "Expecting only fixed indices in variation.")
            return vval, vcomp

        def compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp):
            # Apply gradients directly to argument vval,
            # and get the right indexed scalar component(s)
            kk = indices(ngrads)
            Dvkk = apply_grads(vval)[vcomp+kk]
            # Place scalar component(s) Dvkk into the right tensor positions
            if wshape:
                Ejj, jj = unit_indexed_tensor(wshape, wcomp)
            else:
                Ejj, jj = 1, ()
            gprimeterm = as_tensor(Ejj*Dvkk, jj+kk)
            return gprimeterm

        # Accumulate contributions from variations in different components
        for (w, v) in zip(self._w, self._v):

            # Analyse differentiation variable coefficient
            if isinstance(w, FormArgument):
                if not w == o: continue
                wshape = w.ufl_shape

                if isinstance(v, FormArgument):
                    # Case: d/dt [w + t v]
                    return (g, apply_grads(v))

                elif isinstance(v, ListTensor):
                    # Case: d/dt [w + t <...,v,...>]
                    for wcomp, vsub in unwrap_list_tensor(v):
                        if not isinstance(vsub, Zero):
                            vval, vcomp = analyse_variation_argument(vsub)
                            gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

                else:
                    ufl_assert(wshape == (), "Expecting scalar coefficient in this branch.")
                    # Case: d/dt [w + t v[...]]
                    wval, wcomp = w, ()

                    vval, vcomp = analyse_variation_argument(v)
                    gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            elif isinstance(w, Indexed): # This path is tested in unit tests, but not actually used?
                # Case: d/dt [w[...] + t v[...]]
                # Case: d/dt [w[...] + t v]
                wval, wcomp = w.ufl_operands
                if not wval == o: continue
                assert isinstance(wval, FormArgument)
                ufl_assert(all(isinstance(k, FixedIndex) for k in wcomp),
                           "Expecting only fixed indices in differentiation variable.")
                wshape = wval.ufl_shape

                vval, vcomp = analyse_variation_argument(v)
                gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            else:
                error("Expecting coefficient or component of coefficient.")

        # FIXME: Handle other coefficient derivatives: oprimes = self._cd.get(o)

        if 0:
            oprimes = self._cd.get(o)
            if oprimes is None:
                if self._cd:
                    # TODO: Make it possible to silence this message in particular?
                    #       It may be good to have for debugging...
                    warning("Assuming d{%s}/d{%s} = 0." % (o, self._w))
            else:
                # Make sure we have a tuple to match the self._v tuple
                if not isinstance(oprimes, tuple):
                    oprimes = (oprimes,)
                    ufl_assert(len(oprimes) == len(self._v), "Got a tuple of arguments, "+\
                                   "expecting a matching tuple of coefficient derivatives.")

                # Compute dg/dw_j = dg/dw_h : v.
                # Since we may actually have a tuple of oprimes and vs in a
                # 'mixed' space, sum over them all to get the complete inner
                # product. Using indices to define a non-compound inner product.
                for (oprime, v) in zip(oprimes, self._v):
                    error("FIXME: Figure out how to do this with ngrads")
                    so, oi = as_scalar(oprime)
                    rv = len(v.ufl_shape)
                    oi1 = oi[:-rv]
                    oi2 = oi[-rv:]
                    prod = so*v[oi2]
                    if oi1:
                        gprimesum += as_tensor(prod, oi1)
                    else:
                        gprimesum += prod

        return (g, gprimesum)

    def variable(self, o):
        # Check variable cache to reuse previously transformed variable if possible
        e, l = o.ufl_operands
        c = self._variable_cache.get(l)
        if c is not None:
            return c

        # Visit the expression our variable represents
        e2, ep = self.visit(e)
        op = ep

        # If the expression is not the same, reconstruct Variable object
        o = self.reuse_if_possible(o, e2, l)

        # Recreate Variable (with same label) and cache it
        c = (o, op)
        self._variable_cache[l] = c
        return c

def compute_grad_forward_ad(f, geometric_dimension):
    alg = GradAD(geometric_dimension)
    e, ediff = alg.visit(f)
    return ediff

def compute_variable_forward_ad(f, v):
    alg = VariableAD(v)
    e, ediff = alg.visit(f)
    return ediff

def compute_coefficient_forward_ad(f, w, v, cd):
    alg = CoefficientAD(w, v, cd)
    e, ediff = alg.visit(f)
    return ediff

def apply_nested_forward_ad(expr):
    if expr._ufl_is_terminal_:
        # A terminal needs no differentiation applied
        return expr
    elif not isinstance(expr, Derivative):
        # Apply AD recursively to children
        preops = expr.ufl_operands
        postops = tuple(apply_nested_forward_ad(o) for o in preops)
        # Reconstruct if necessary
        need_reconstruct = not (preops == postops) # FIXME: Is this efficient? O(n)?
        if need_reconstruct:
            expr = expr._ufl_expr_reconstruct_(*postops)
        return expr
    elif isinstance(expr, Grad):
        # Apply AD recursively to children
        f, = expr.ufl_operands
        f = apply_nested_forward_ad(f)
        # Apply Grad-specialized AD to expanded child
        gdim = expr.ufl_shape[-1]
        return compute_grad_forward_ad(f, gdim)
    elif isinstance(expr, VariableDerivative):
        # Apply AD recursively to children
        f, v = expr.ufl_operands
        f = apply_nested_forward_ad(f)
        # Apply Variable-specialized AD to expanded child
        return compute_variable_forward_ad(f, v)
    elif isinstance(expr, CoefficientDerivative):
        # Apply AD recursively to children
        f, w, v, cd = expr.ufl_operands
        f = apply_nested_forward_ad(f)
        # Apply Coefficient-specialized AD to expanded child
        return compute_coefficient_forward_ad(f, w, v, cd)
    else:
        error("Unknown type.")

# TODO: We could expand only the compound objects that have no rule
#       before differentiating, to allow the AD to work on a coarser graph
class UnusedADRules(object):

    def _variable_derivative(self, o, f, v):
        f, fp = f
        v, vp = v
        ufl_assert(isinstance(vp, Zero), "TODO: What happens if vp != 0, i.e. v depends the differentiation variable?")
        # Are there any issues with indices here? Not sure, think through it...
        oprime = o._ufl_expr_reconstruct_(fp, v)
        return (o, oprime)

    # --- Tensor algebra (compound types)

    def outer(self, o, a, b):
        a, ap = a
        b, bp = b
        return (o, outer(ap, b) + outer(a, bp)) # FIXME: Not valid for derivatives w.r.t. nonscalar variables!

    def inner(self, o, a, b):
        a, ap = a
        b, bp = b
        # FIXME: Rewrite with index notation (if necessary, depends on shapes)
        # NB! Using b : ap because derivative axis should be
        # last, in case of nonscalar differentiation variable!
        return (o, inner(b, ap) + inner(a, bp)) # FIXME: Not correct, inner requires equal shapes!

    def dot(self, o, a, b):
        a, ap = a
        b, bp = b
        # FIXME: Rewrite with index notation (if necessary, depends on shapes)
        # NB! Using b . ap because derivative axis should be
        # last, in case of nonscalar differentiation variable!
        return (o, dot(b, ap) + dot(a, bp))

    def commute(self, o, a):
        "This should work for all single argument operators that commute with d/dw with w scalar."
        aprime = a[1]
        return (o, o._ufl_expr_reconstruct_(aprime))

    # FIXME: Not true for derivatives w.r.t. nonscalar variables...
    transposed = commute
    trace = commute
    deviatoric = commute

    # --- Compound differential operators, probably do not want...

    # FIXME: nabla_div, nabla_grad
    div  = commute
    curl = commute
    def grad(self, o, a):
        a, aprime = a
        if is_cellwise_constant(aprime):
            oprime = self._make_zero_diff(o)
        else:
            oprime = o._ufl_expr_reconstruct_(aprime)
        return (o, oprime)

class UnimplementedADRules(object):

    def cross(self, o, a, b):
        error("Derivative of cross product not implemented, apply expand_compounds before AD.")
        u, up = a
        v, vp = b
        #oprime = ...
        return (o, oprime)

    def determinant(self, o, a):
        """FIXME: Some possible rules:

        d detA / dv = detA * tr(inv(A) * dA/dv)

        or

        d detA / d row0 = cross(row1, row2)
        d detA / d row1 = cross(row2, row0)
        d detA / d row2 = cross(row0, row1)

        i.e.

        d detA / d A = [cross(row1, row2), cross(row2, row0), cross(row0, row1)] # or transposed or something
        """
        error("Derivative of determinant not implemented, apply expand_compounds before AD.")
        A, Ap = a
        #oprime = ...
        return (o, oprime)

    def cofactor(self, o, a):
        error("Derivative of cofactor not implemented, apply expand_compounds before AD.")
        A, Ap = a
        #cofacA_prime = detA_prime*Ainv + detA*Ainv_prime
        #oprime = ...
        return (o, oprime)

    def inverse(self, o, a):
        """Derivation:
        0 = d/dx [Ainv*A] = Ainv' * A + Ainv * A'
        Ainv' * A = - Ainv * A'
        Ainv' = - Ainv * A' * Ainv

        K J = I
        d/dv[r] (K[i,j] J[j,k]) = d/dv[r] K[i,j] J[j,k] + K[i,j] d/dv[r] J[j,k] = 0
        d/dv[r] K[i,j] J[j,k] = -K[i,j] d/dv[r] J[j,k]
        d/dv[r] K[i,j] J[j,k] K[k,s] = -K[i,j] d/dv[r] J[j,k] K[k,s]
        d/dv[r] K[i,j] I[j,s] = -K[i,j] (d/dv[r] J[j,k]) K[k,s]
        d/dv[r] K[i,s] = -K[i,j] (d/dv[r] J[j,k]) K[k,s]
        """
        A, Ap = a
        return (o, -o*Ap*o) # FIXME: Need reshaping move derivative axis to the end of this expression
