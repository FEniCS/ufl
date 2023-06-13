"""This module contains the apply_derivatives algorithm which computes the derivatives of a form of expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import warnings
from collections import defaultdict
from math import pi

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.replace_derivative_nodes import replace_derivative_nodes
from ufl.checks import is_cellwise_constant
from ufl.classes import (Coefficient, ComponentTensor, Conj, ConstantValue,
                         ExprList, ExprMapping, FloatValue, FormArgument, Grad,
                         Identity, Imag, Indexed, IndexSum, JacobianInverse,
                         ListTensor, Product, Real, ReferenceGrad,
                         ReferenceValue, SpatialCoordinate, Sum, Variable,
                         Zero)
from ufl.constantvalue import is_true_ufl_scalar, is_ufl_scalar
from ufl.core.expr import ufl_err_str
from ufl.core.multiindex import FixedIndex, MultiIndex, indices
from ufl.core.terminal import Terminal
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.differentiation import CoordinateDerivative, BaseFormOperatorDerivative
from ufl.domain import extract_unique_domain
from ufl.operators import (bessel_I, bessel_J, bessel_K, bessel_Y, cell_avg,
                           conditional, cos, cosh, exp, facet_avg, ln, sign,
                           sin, sinh, sqrt)
from ufl.tensors import (as_scalar, as_scalars, as_tensor, unit_indexed_tensor,
                         unwrap_list_tensor)

from ufl.argument import BaseArgument
from ufl.action import Action
from ufl.form import Form, ZeroBaseForm
from ufl.core.base_form_operator import BaseFormOperator
# TODO: Add more rulesets?
# - DivRuleset
# - CurlRuleset
# - ReferenceGradRuleset
# - ReferenceDivRuleset


# Set this to True to enable previously default workaround
# for bug in FFC handling of conditionals, uflacs does not
# have this bug.
CONDITIONAL_WORKAROUND = False


class GenericDerivativeRuleset(MultiFunction):
    def __init__(self, var_shape):
        MultiFunction.__init__(self)
        self._var_shape = var_shape

    # --- Error checking for missing handlers and unexpected types

    def expr(self, o):
        raise ValueError(f"Missing differentiation handler for type {o._ufl_class_.__name__}. Have you added a new type?")

    def unexpected(self, o):
        raise ValueError(f"Unexpected type {o._ufl_class_.__name__} in AD rules.")

    def override(self, o):
        raise ValueError(f"Type {o._ufl_class_.__name__} must be overridden in specialized AD rule set.")

    def derivative(self, o):
        raise ValueError(f"Unhandled derivative type {o._ufl_class_.__name__}, nested differentiation has failed.")

    # --- Some types just don't have any derivative, this is just to
    # --- make algorithm structure generic

    def non_differentiable_terminal(self, o):
        "Labels and indices are not differentiable. It's convenient to return the non-differentiated object."
        return o
    label = non_differentiable_terminal
    multi_index = non_differentiable_terminal

    # --- Helper functions for creating zeros with the right shapes

    def independent_terminal(self, o):
        "Return a zero with the right shape for terminals independent of differentiation variable."
        return Zero(o.ufl_shape + self._var_shape)

    def independent_operator(self, o):
        "Return a zero with the right shape and indices for operators independent of differentiation variable."
        return Zero(o.ufl_shape + self._var_shape, o.ufl_free_indices, o.ufl_index_dimensions)

    # --- All derivatives need to define grad and averaging

    grad = override
    cell_avg = override
    facet_avg = override

    # --- Default rules for terminals

    # Literals are by definition independent of any differentiation variable
    constant_value = independent_terminal

    # Constants are independent of any differentiation
    constant = independent_terminal

    # Rules for form arguments must be specified in specialized rule set
    form_argument = override

    # Rules for geometric quantities must be specified in specialized rule set
    geometric_quantity = override

    # These types are currently assumed independent, but for non-affine domains
    # this no longer holds and we want to implement rules for them.
    # facet_normal = independent_terminal
    # spatial_coordinate = independent_terminal
    # cell_coordinate = independent_terminal

    # Measures of cell entities, assuming independent although
    # this will not be true for all of these for non-affine domains
    # cell_volume = independent_terminal
    # circumradius = independent_terminal
    # facet_area = independent_terminal
    # cell_surface_area = independent_terminal
    # min_cell_edge_length = independent_terminal
    # max_cell_edge_length = independent_terminal
    # min_facet_edge_length = independent_terminal
    # max_facet_edge_length = independent_terminal

    # Other stuff
    # cell_orientation = independent_terminal
    # quadrature_weigth = independent_terminal

    # These types are currently not expected to show up in AD pass.
    # To make some of these available to the end-user, they need to be
    # implemented here.
    # facet_coordinate = unexpected
    # cell_origin = unexpected
    # facet_origin = unexpected
    # cell_facet_origin = unexpected
    # jacobian = unexpected
    # jacobian_determinant = unexpected
    # jacobian_inverse = unexpected
    # facet_jacobian = unexpected
    # facet_jacobian_determinant = unexpected
    # facet_jacobian_inverse = unexpected
    # cell_facet_jacobian = unexpected
    # cell_facet_jacobian_determinant = unexpected
    # cell_facet_jacobian_inverse = unexpected
    # cell_vertices = unexpected
    # cell_edge_vectors = unexpected
    # facet_edge_vectors = unexpected
    # reference_cell_edge_vectors = unexpected
    # reference_facet_edge_vectors = unexpected
    # cell_normal = unexpected # TODO: Expecting rename
    # cell_normals = unexpected
    # facet_tangents = unexpected
    # cell_tangents = unexpected
    # cell_midpoint = unexpected
    # facet_midpoint = unexpected

    # --- Default rules for operators

    def variable(self, o, df, unused_l):
        return df

    # --- Indexing and component handling

    def indexed(self, o, Ap, ii):  # TODO: (Partially) duplicated in nesting rules
        # Propagate zeros
        if isinstance(Ap, Zero):
            return self.independent_operator(o)

        # Untangle as_tensor(C[kk], jj)[ii] -> C[ll] to simplify
        # resulting expression
        if isinstance(Ap, ComponentTensor):
            B, jj = Ap.ufl_operands
            if isinstance(B, Indexed):
                C, kk = B.ufl_operands
                kk = list(kk)
                if all(j in kk for j in jj):
                    rep = dict(zip(jj, ii))
                    Cind = [rep.get(k, k) for k in kk]
                    expr = Indexed(C, MultiIndex(tuple(Cind)))
                    assert expr.ufl_free_indices == o.ufl_free_indices
                    assert expr.ufl_shape == o.ufl_shape
                    return expr

        # Otherwise a more generic approach
        r = len(Ap.ufl_shape) - len(ii)
        if r:
            kk = indices(r)
            op = Indexed(Ap, MultiIndex(ii.indices() + kk))
            op = as_tensor(op, kk)
        else:
            op = Indexed(Ap, ii)
        return op

    def list_tensor(self, o, *dops):
        return ListTensor(*dops)

    def component_tensor(self, o, Ap, ii):
        if isinstance(Ap, Zero):
            op = self.independent_operator(o)
        else:
            Ap, jj = as_scalar(Ap)
            op = as_tensor(Ap, ii.indices() + jj)
        return op

    # --- Algebra operators

    def index_sum(self, o, Ap, i):
        return IndexSum(Ap, i)

    def sum(self, o, da, db):
        return da + db

    def product(self, o, da, db):
        # Even though arguments to o are scalar, da and db may be
        # tensor valued
        a, b = o.ufl_operands
        (da, db), ii = as_scalars(da, db)
        pa = Product(da, b)
        pb = Product(a, db)
        s = Sum(pa, pb)
        if ii:
            s = as_tensor(s, ii)
        return s

    def division(self, o, fp, gp):
        f, g = o.ufl_operands

        if not is_ufl_scalar(f):
            raise ValueError("Not expecting nonscalar nominator")
        if not is_true_ufl_scalar(g):
            raise ValueError("Not expecting nonscalar denominator")

        # do_df = 1/g
        # do_dg = -h/g
        # op = do_df*fp + do_df*gp
        # op = (fp - o*gp) / g

        # Get o and gp as scalars, multiply, then wrap as a tensor
        # again
        so, oi = as_scalar(o)
        sgp, gi = as_scalar(gp)
        o_gp = so * sgp
        if oi or gi:
            o_gp = as_tensor(o_gp, oi + gi)
        op = (fp - o_gp) / g

        return op

    def power(self, o, fp, gp):
        f, g = o.ufl_operands

        if not is_true_ufl_scalar(f):
            raise ValueError("Expecting scalar expression f in f**g.")
        if not is_true_ufl_scalar(g):
            raise ValueError("Expecting scalar expression g in f**g.")

        # Derivation of the general case: o = f(x)**g(x)
        # do/df  = g * f**(g-1) = g / f * o
        # do/dg  = ln(f) * f**g = ln(f) * o
        # do/df * df + do/dg * dg = o * (g / f * df + ln(f) * dg)

        if isinstance(gp, Zero):
            # This probably produces better results for the common
            # case of f**constant
            op = fp * g * f**(g - 1)
        else:
            # Note: This produces expressions like (1/w)*w**5 instead of w**4
            # op = o * (fp * g / f + gp * ln(f)) # This reuses o
            op = f**(g - 1) * (g * fp + f * ln(f) * gp)  # This gives better accuracy in dolfin integration test

        # Example: d/dx[x**(x**3)]:
        # f = x
        # g = x**3
        # df = 1
        # dg = 3*x**2
        # op1 = o * (fp * g / f + gp * ln(f))
        #     = x**(x**3)   * (x**3/x + 3*x**2*ln(x))
        # op2 = f**(g-1) * (g*fp + f*ln(f)*gp)
        #     = x**(x**3-1) * (x**3 + x*3*x**2*ln(x))

        return op

    def abs(self, o, df):
        f, = o.ufl_operands
        # return conditional(eq(f, 0), 0, Product(sign(f), df)) abs is
        # not complex differentiable, so we workaround the case of a
        # real F in complex mode by defensively casting to real inside
        # the sign.
        return sign(Real(f)) * df

    # --- Complex algebra

    def conj(self, o, df):
        return Conj(df)

    def real(self, o, df):
        return Real(df)

    def imag(self, o, df):
        return Imag(df)

    # --- Mathfunctions

    def math_function(self, o, df):
        # FIXME: Introduce a UserOperator type instead of this hack
        # and define user derivative() function properly
        if hasattr(o, 'derivative'):
            f, = o.ufl_operands
            return df * o.derivative()
        raise ValueError("Unknown math function.")

    def sqrt(self, o, fp):
        return fp / (2 * o)

    def exp(self, o, fp):
        return fp * o

    def ln(self, o, fp):
        f, = o.ufl_operands
        if isinstance(f, Zero):
            raise ZeroDivisionError()
        return fp / f

    def cos(self, o, fp):
        f, = o.ufl_operands
        return fp * -sin(f)

    def sin(self, o, fp):
        f, = o.ufl_operands
        return fp * cos(f)

    def tan(self, o, fp):
        f, = o.ufl_operands
        return 2.0 * fp / (cos(2.0 * f) + 1.0)

    def cosh(self, o, fp):
        f, = o.ufl_operands
        return fp * sinh(f)

    def sinh(self, o, fp):
        f, = o.ufl_operands
        return fp * cosh(f)

    def tanh(self, o, fp):
        f, = o.ufl_operands

        def sech(y):
            return (2.0 * cosh(y)) / (cosh(2.0 * y) + 1.0)
        return fp * sech(f)**2

    def acos(self, o, fp):
        f, = o.ufl_operands
        return -fp / sqrt(1.0 - f**2)

    def asin(self, o, fp):
        f, = o.ufl_operands
        return fp / sqrt(1.0 - f**2)

    def atan(self, o, fp):
        f, = o.ufl_operands
        return fp / (1.0 + f**2)

    def atan_2(self, o, fp, gp):
        f, g = o.ufl_operands
        return (g * fp - f * gp) / (f**2 + g**2)

    def erf(self, o, fp):
        f, = o.ufl_operands
        return fp * (2.0 / sqrt(pi) * exp(-f**2))

    # --- Bessel functions

    def bessel_j(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = -bessel_J(1, f)
        else:
            op = 0.5 * (bessel_J(nu - 1, f) - bessel_J(nu + 1, f))
        return op * fp

    def bessel_y(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = -bessel_Y(1, f)
        else:
            op = 0.5 * (bessel_Y(nu - 1, f) - bessel_Y(nu + 1, f))
        return op * fp

    def bessel_i(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = bessel_I(1, f)
        else:
            op = 0.5 * (bessel_I(nu - 1, f) + bessel_I(nu + 1, f))
        return op * fp

    def bessel_k(self, o, nup, fp):
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError("Differentiation of bessel function w.r.t. nu is not supported.")

        if isinstance(nu, Zero):
            op = -bessel_K(1, f)
        else:
            op = -0.5 * (bessel_K(nu - 1, f) + bessel_K(nu + 1, f))
        return op * fp

    # --- Restrictions

    def restricted(self, o, fp):
        # Restriction and differentiation commutes
        if isinstance(fp, ConstantValue):
            return fp  # TODO: Add simplification to Restricted instead?
        else:
            return fp(o._side)  # (f+-)' == (f')+-

    # --- Conditionals

    def binary_condition(self, o, dl, dr):
        # Should not be used anywhere...
        return None

    def not_condition(self, o, c):
        # Should not be used anywhere...
        return None

    def conditional(self, o, unused_dc, dt, df):
        global CONDITIONAL_WORKAROUND
        if isinstance(dt, Zero) and isinstance(df, Zero):
            # Assuming dt and df have the same indices here, which
            # should be the case
            return dt
        elif CONDITIONAL_WORKAROUND:
            # Placing t[1],f[1] outside here to avoid getting
            # arguments inside conditionals.  This will fail when dt
            # or df become NaN or Inf in floating point computations!
            c = o.ufl_operands[0]
            dc = conditional(c, 1, 0)
            return dc * dt + (1.0 - dc) * df
        else:
            # Not placing t[1],f[1] outside, allowing arguments inside
            # conditionals.  This will make legacy ffc fail, but
            # should work with uflacs.
            c = o.ufl_operands[0]
            return conditional(c, dt, df)

    def max_value(self, o, df, dg):
        # d/dx max(f, g) =
        # f > g: df/dx
        # f < g: dg/dx
        # Placing df,dg outside here to avoid getting arguments inside
        # conditionals
        f, g = o.ufl_operands
        dc = conditional(f > g, 1, 0)
        return dc * df + (1.0 - dc) * dg

    def min_value(self, o, df, dg):
        # d/dx min(f, g) =
        #  f < g: df/dx
        #  else: dg/dx
        #  Placing df,dg outside here to avoid getting arguments
        #  inside conditionals
        f, g = o.ufl_operands
        dc = conditional(f < g, 1, 0)
        return dc * df + (1.0 - dc) * dg


class GradRuleset(GenericDerivativeRuleset):
    def __init__(self, geometric_dimension):
        GenericDerivativeRuleset.__init__(self, var_shape=(geometric_dimension,))
        self._Id = Identity(geometric_dimension)

    # --- Specialized rules for geometric quantities

    def geometric_quantity(self, o):
        """Default for geometric quantities is do/dx = 0 if piecewise constant,
        otherwise transform derivatives to reference derivatives.
        Override for specific types if other behaviour is needed."""
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        else:
            domain = extract_unique_domain(o)
            K = JacobianInverse(domain)
            Do = grad_to_reference_grad(o, K)
            return Do

    def jacobian_inverse(self, o):
        # grad(K) == K_ji rgrad(K)_rj
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        if not o._ufl_is_terminal_:
            raise ValueError("ReferenceValue can only wrap a terminal")
        Do = grad_to_reference_grad(o, o)
        return Do

    # TODO: Add more explicit geometry type handlers here, with
    # non-affine domains several should be non-zero.

    def spatial_coordinate(self, o):
        "dx/dx = I"
        return self._Id

    def cell_coordinate(self, o):
        "dX/dx = inv(dx/dX) = inv(J) = K"
        # FIXME: Is this true for manifolds? What about orientation?
        return JacobianInverse(extract_unique_domain(o))

    # --- Specialized rules for form arguments

    def interp(self, o):
        # Push the grad through the operator is not legal in most cases:
        #    -> Not enouth regularity for chain rule to hold!
        # By the time we evaluate `grad(o)`, `o` (Interp) will have been assembled and substituted by its output.
        return Grad(o)

    def coefficient(self, o):
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        return Grad(o)

    def argument(self, o):
        # TODO: Enable this after fixing issue#13, unless we move
        # simplificat ion to a separate stage?
        # if is_cellwise_constant(o):
        #     # Collapse gradient of cellwise constant function to zero
        #     # TODO: Missing this type
        #     return AnnotatedZero(o.ufl_shape + self._var_shape, arguments=(o,))
        return Grad(o)

    # --- Rules for values or derivatives in reference frame

    def reference_value(self, o):
        # grad(o) == grad(rv(f)) -> K_ji*rgrad(rv(f))_rj
        f = o.ufl_operands[0]
        if f.ufl_element().mapping() == "physical":
            # TODO: Do we need to be more careful for immersed things?
            return ReferenceGrad(o)

        if not f._ufl_is_terminal_:
            raise ValueError("ReferenceValue can only wrap a terminal")
        domain = extract_unique_domain(f)
        K = JacobianInverse(domain)
        Do = grad_to_reference_grad(o, K)
        return Do

    def reference_grad(self, o):
        # grad(o) == grad(rgrad(rv(f))) -> K_ji*rgrad(rgrad(rv(f)))_rj
        f = o.ufl_operands[0]

        valid_operand = f._ufl_is_in_reference_frame_ or isinstance(f, (JacobianInverse, SpatialCoordinate))
        if not valid_operand:
            raise ValueError("ReferenceGrad can only wrap a reference frame type!")
        domain = extract_unique_domain(f)
        K = JacobianInverse(domain)
        Do = grad_to_reference_grad(o, K)
        return Do

    # --- Nesting of gradients

    def grad(self, o):
        "Represent grad(grad(f)) as Grad(Grad(f))."

        # Check that o is a "differential terminal"
        if not isinstance(o.ufl_operands[0], (Grad, Terminal)):
            raise ValueError("Expecting only grads applied to a terminal.")

        return Grad(o)

    def _grad(self, o):
        pass
        # TODO: Not sure how to detect that gradient of f is cellwise constant.
        #       Can we trust element degrees?
        # if is_cellwise_constant(o):
        #     return self.terminal(o)
        # TODO: Maybe we can ask "f.has_derivatives_of_order(n)" to check
        #       if we should make a zero here?
        # 1) n = count number of Grads, get f
        # 2) if not f.has_derivatives(n): return zero(...)

    cell_avg = GenericDerivativeRuleset.independent_operator
    facet_avg = GenericDerivativeRuleset.independent_operator


def grad_to_reference_grad(o, K):
    """Relates grad(o) to reference_grad(o) using the Jacobian inverse.

    Args
    ----
    o: Operand
    K: Jacobian inverse

    Returns
    -------
    Do: grad(o) written in terms of reference_grad(o) and K

    """
    r = indices(len(o.ufl_shape))
    i, j = indices(2)
    # grad(o) == K_ji rgrad(o)_rj
    Do = as_tensor(K[j, i] * ReferenceGrad(o)[r + (j,)], r + (i,))
    return Do


class ReferenceGradRuleset(GenericDerivativeRuleset):
    def __init__(self, topological_dimension):
        GenericDerivativeRuleset.__init__(self,
                                          var_shape=(topological_dimension,))
        self._Id = Identity(topological_dimension)

    # --- Specialized rules for geometric quantities

    def geometric_quantity(self, o):
        "dg/dX = 0 if piecewise constant, otherwise ReferenceGrad(g)"
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        else:
            # TODO: Which types does this involve? I don't think the
            # form compilers will handle this.
            return ReferenceGrad(o)

    def spatial_coordinate(self, o):
        "dx/dX = J"
        # Don't convert back to J, otherwise we get in a loop
        return ReferenceGrad(o)

    def cell_coordinate(self, o):
        "dX/dX = I"
        return self._Id

    # TODO: Add more geometry types here, with non-affine domains
    # several should be non-zero.

    # --- Specialized rules for form arguments

    def reference_value(self, o):
        if not o.ufl_operands[0]._ufl_is_terminal_:
            raise ValueError("ReferenceValue can only wrap a terminal")
        return ReferenceGrad(o)

    def coefficient(self, o):
        raise ValueError("Coefficient should be wrapped in ReferenceValue by now")

    def argument(self, o):
        raise ValueError("Argument should be wrapped in ReferenceValue by now")

    # --- Nesting of gradients

    def grad(self, o):
        raise ValueError(f"Grad should have been transformed by this point, but got {type(o).__name__}.")

    def reference_grad(self, o):
        "Represent ref_grad(ref_grad(f)) as RefGrad(RefGrad(f))."
        # Check that o is a "differential terminal"
        if not isinstance(o.ufl_operands[0],
                          (ReferenceGrad, ReferenceValue, Terminal)):
            raise ValueError("Expecting only grads applied to a terminal.")
        return ReferenceGrad(o)

    cell_avg = GenericDerivativeRuleset.independent_operator
    facet_avg = GenericDerivativeRuleset.independent_operator


class VariableRuleset(GenericDerivativeRuleset):
    def __init__(self, var):
        GenericDerivativeRuleset.__init__(self, var_shape=var.ufl_shape)
        if var.ufl_free_indices:
            raise ValueError("Differentiation variable cannot have free indices.")
        self._variable = var
        self._Id = self._make_identity(self._var_shape)

    def _make_identity(self, sh):
        "Create a higher order identity tensor to represent dv/dv."
        res = None
        if sh == ():
            # Scalar dv/dv is scalar
            return FloatValue(1.0)
        elif len(sh) == 1:
            # Vector v makes dv/dv the identity matrix
            return Identity(sh[0])
        else:
            # TODO: Add a type for this higher order identity?
            # II[i0,i1,i2,j0,j1,j2] = 1 if all((i0==j0, i1==j1, i2==j2)) else 0
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

    # Explicitly defining dg/dw == 0
    geometric_quantity = GenericDerivativeRuleset.independent_terminal

    # Explicitly defining da/dw == 0
    argument = GenericDerivativeRuleset.independent_terminal

    # def _argument(self, o):
    #    return AnnotatedZero(o.ufl_shape + self._var_shape, arguments=(o,))  # TODO: Missing this type

    def coefficient(self, o):
        """df/dv = Id if v is f else 0.

        Note that if v = variable(f), df/dv is still 0,
        but if v == f, i.e. isinstance(v, Coefficient) == True,
        then df/dv == df/df = Id.
        """
        v = self._variable
        if isinstance(v, Coefficient) and o == v:
            # dv/dv = identity of rank 2*rank(v)
            return self._Id
        else:
            # df/v = 0
            return self.independent_terminal(o)

    def variable(self, o, df, l):
        v = self._variable
        if isinstance(v, Variable) and v.label() == l:
            # dv/dv = identity of rank 2*rank(v)
            return self._Id
        else:
            # df/v = df
            return df

    def grad(self, o):
        "Variable derivative of a gradient of a terminal must be 0."
        # Check that o is a "differential terminal"
        if not isinstance(o.ufl_operands[0], (Grad, Terminal)):
            raise ValueError("Expecting only grads applied to a terminal.")
        return self.independent_terminal(o)

    # --- Rules for values or derivatives in reference frame

    def reference_value(self, o):
        # d/dv(o) == d/dv(rv(f)) = 0 if v is not f, or rv(dv/df)
        v = self._variable
        if isinstance(v, Coefficient) and o.ufl_operands[0] == v:
            if v.ufl_element().mapping() != "identity":
                # FIXME: This is a bit tricky, instead of Identity it is
                #   actually inverse(transform), or we should rather not
                #   convert to reference frame in the first place
                raise ValueError(
                    "Missing implementation: To handle derivatives of rv(f) w.r.t. f for "
                    "mapped elements, rewriting to reference frame should not happen first...")
            # dv/dv = identity of rank 2*rank(v)
            return self._Id
        else:
            # df/v = 0
            return self.independent_terminal(o)

    def reference_grad(self, o):
        "Variable derivative of a gradient of a terminal must be 0."
        if not isinstance(o.ufl_operands[0],
                          (ReferenceGrad, ReferenceValue)):
            raise ValueError("Unexpected argument to reference_grad.")
        return self.independent_terminal(o)

    cell_avg = GenericDerivativeRuleset.independent_operator
    facet_avg = GenericDerivativeRuleset.independent_operator


class GateauxDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AFD (Automatic Functional Differentiation) to expression.

    Implements rules for the Gateaux derivative D_w[v](...) defined as

        D_w[v](e) = d/dtau e(w+tau v)|tau=0

    """

    def __init__(self, coefficients, arguments, coefficient_derivatives):
        GenericDerivativeRuleset.__init__(self, var_shape=())

        # Type checking
        if not isinstance(coefficients, ExprList):
            raise ValueError("Expecting a ExprList of coefficients.")
        if not isinstance(arguments, ExprList):
            raise ValueError("Expecting a ExprList of arguments.")
        if not isinstance(coefficient_derivatives, ExprMapping):
            raise ValueError("Expecting a coefficient-coefficient ExprMapping.")

        # The coefficient(s) to differentiate w.r.t. and the
        # argument(s) s.t. D_w[v](e) = d/dtau e(w+tau v)|tau=0
        self._w = coefficients.ufl_operands
        self._v = arguments.ufl_operands
        self._w2v = {w: v for w, v in zip(self._w, self._v)}

        # Build more convenient dict {f: df/dw} for each coefficient f
        # where df/dw is nonzero
        cd = coefficient_derivatives.ufl_operands
        self._cd = {cd[2 * i]: cd[2 * i + 1] for i in range(len(cd) // 2)}

        # Record the operations delayed to the derivative expansion phase:
        # Example: dN(u)/du where `N` is an ExternalOperator and `u` a Coefficient
        self.pending_operations = (coefficients, {'arguments': arguments, 'coefficient_derivatives': coefficient_derivatives})

    # Explicitly defining dg/dw == 0
    geometric_quantity = GenericDerivativeRuleset.independent_terminal

    def cell_avg(self, o, fp):
        # Cell average of a single function and differentiation
        # commutes, D_f[v](cell_avg(f)) = cell_avg(v)
        return cell_avg(fp)

    def facet_avg(self, o, fp):
        # Facet average of a single function and differentiation
        # commutes, D_f[v](facet_avg(f)) = facet_avg(v)
        return facet_avg(fp)

    # Explicitly defining da/dw == 0
    argument = GenericDerivativeRuleset.independent_terminal

    def coefficient(self, o):
        # Define dw/dw := d/ds [w + s v] = v

        # Return corresponding argument if we can find o among w
        do = self._w2v.get(o)
        if do is not None:
            return do

        # Look for o among coefficient derivatives
        dos = self._cd.get(o)
        if dos is None:
            # If o is not among coefficient derivatives, return
            # do/dw=0
            do = Zero(o.ufl_shape)
            return do
        else:
            # Compute do/dw_j = do/dw_h : v.
            # Since we may actually have a tuple of oprimes and vs in a
            # 'mixed' space, sum over them all to get the complete inner
            # product. Using indices to define a non-compound inner product.

            # Example:
            # (f:g) -> (dfdu:v):g + f:(dgdu:v)
            # shape(dfdu) == shape(f) + shape(v)
            # shape(f) == shape(g) == shape(dfdu : v)

            # Make sure we have a tuple to match the self._v tuple
            if not isinstance(dos, tuple):
                dos = (dos,)
            if len(dos) != len(self._v):
                raise ValueError("Got a tuple of arguments, expecting a matching tuple of coefficient derivatives.")
            dosum = Zero(o.ufl_shape)
            for do, v in zip(dos, self._v):
                so, oi = as_scalar(do)
                rv = len(v.ufl_shape)
                oi1 = oi[:-rv]
                oi2 = oi[-rv:]
                prod = so * v[oi2]
                if oi1:
                    dosum += as_tensor(prod, oi1)
                else:
                    dosum += prod
            return dosum

    def reference_value(self, o):
        raise NotImplementedError("Currently no support for ReferenceValue in CoefficientDerivative.")
        # TODO: This is implementable for regular derivative(M(f),f,v)
        #       but too messy if customized coefficient derivative
        #       relations are given by the user.  We would only need
        #       this to allow the user to write
        #       derivative(...ReferenceValue...,...).
        # f, = o.ufl_operands
        # if not f._ufl_is_terminal_:
        #     raise ValueError("ReferenceValue can only wrap terminals directly.")
        # FIXME: check all cases like in coefficient
        # if f is w:
        #     # FIXME: requires that v is an Argument with the same element mapping!
        #     return ReferenceValue(v)
        # else:
        #     return self.independent_terminal(o)

    def reference_grad(self, o):
        raise NotImplementedError("Currently no support for ReferenceGrad in CoefficientDerivative.")
        # TODO: This is implementable for regular derivative(M(f),f,v)
        #       but too messy if customized coefficient derivative
        #       relations are given by the user.  We would only need
        #       this to allow the user to write
        #       derivative(...ReferenceValue...,...).

    def grad(self, g):
        # If we hit this type, it has already been propagated to a
        # coefficient (or grad of a coefficient) or a base form operator, # FIXME: Assert
        # this!  so we need to take the gradient of the variation or
        # return zero.  Complications occur when dealing with
        # derivatives w.r.t. single components...

        # Figure out how many gradients are around the inner terminal
        ngrads = 0
        o = g
        while isinstance(o, Grad):
            o, = o.ufl_operands
            ngrads += 1
        # `grad(N)` where N is a BaseFormOperator is treated as if `N` was a Coefficient.
        if not isinstance(o, (FormArgument, BaseFormOperator)):
            raise ValueError(f"Expecting gradient of a FormArgument, not {ufl_err_str(o)}.")

        def apply_grads(f):
            for i in range(ngrads):
                f = Grad(f)
            return f

        # Find o among all w without any indexing, which makes this
        # easy
        for (w, v) in zip(self._w, self._v):
            if o == w and isinstance(v, FormArgument):
                # Case: d/dt [w + t v]
                return apply_grads(v)

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
                raise ValueError("Expecting argument or component of argument.")
            if not all(isinstance(k, FixedIndex) for k in vcomp):
                raise ValueError("Expecting only fixed indices in variation.")
            return vval, vcomp

        def compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp):
            # Apply gradients directly to argument vval, and get the
            # right indexed scalar component(s)
            kk = indices(ngrads)
            Dvkk = apply_grads(vval)[vcomp + kk]
            # Place scalar component(s) Dvkk into the right tensor
            # positions
            if wshape:
                Ejj, jj = unit_indexed_tensor(wshape, wcomp)
            else:
                Ejj, jj = 1, ()
            gprimeterm = as_tensor(Ejj * Dvkk, jj + kk)
            return gprimeterm

        # Accumulate contributions from variations in different
        # components
        for (w, v) in zip(self._w, self._v):

            # -- Analyse differentiation variable coefficient -- #

            # Can differentiate a Form wrt a BaseFormOperator
            if isinstance(w, (FormArgument, BaseFormOperator)):
                if not w == o:
                    continue
                wshape = w.ufl_shape

                if isinstance(v, FormArgument):
                    # Case: d/dt [w + t v]
                    return apply_grads(v)

                elif isinstance(v, ListTensor):
                    # Case: d/dt [w + t <...,v,...>]
                    for wcomp, vsub in unwrap_list_tensor(v):
                        if not isinstance(vsub, Zero):
                            vval, vcomp = analyse_variation_argument(vsub)
                            gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

                else:
                    if wshape != ():
                        raise ValueError("Expecting scalar coefficient in this branch.")
                    # Case: d/dt [w + t v[...]]
                    wval, wcomp = w, ()

                    vval, vcomp = analyse_variation_argument(v)
                    gprimesum = gprimesum + compute_gprimeterm(ngrads, vval,
                                                               vcomp, wshape,
                                                               wcomp)

            elif isinstance(w, Indexed):  # This path is tested in unit tests, but not actually used?
                # Case: d/dt [w[...] + t v[...]]
                # Case: d/dt [w[...] + t v]
                wval, wcomp = w.ufl_operands
                if not wval == o:
                    continue
                assert isinstance(wval, FormArgument)
                if not all(isinstance(k, FixedIndex) for k in wcomp):
                    raise ValueError("Expecting only fixed indices in differentiation variable.")
                wshape = wval.ufl_shape

                vval, vcomp = analyse_variation_argument(v)
                gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            else:
                raise ValueError("Expecting coefficient or component of coefficient.")

        # FIXME: Handle other coefficient derivatives: oprimes =
        # self._cd.get(o)

        if 0:
            oprimes = self._cd.get(o)
            if oprimes is None:
                if self._cd:
                    # TODO: Make it possible to silence this message
                    #       in particular?  It may be good to have for
                    #       debugging...
                    warnings.warn("Assuming d{%s}/d{%s} = 0." % (o, self._w))
            else:
                # Make sure we have a tuple to match the self._v tuple
                if not isinstance(oprimes, tuple):
                    oprimes = (oprimes,)
                    if len(oprimes) != len(self._v):
                        raise ValueError(
                            "Got a tuple of arguments, expecting a"
                            " matching tuple of coefficient derivatives.")

                # Compute dg/dw_j = dg/dw_h : v.
                # Since we may actually have a tuple of oprimes and vs
                # in a 'mixed' space, sum over them all to get the
                # complete inner product. Using indices to define a
                # non-compound inner product.
                for (oprime, v) in zip(oprimes, self._v):
                    raise NotImplementedError("FIXME: Figure out how to do this with ngrads")
                    so, oi = as_scalar(oprime)
                    rv = len(v.ufl_shape)
                    oi1 = oi[:-rv]
                    oi2 = oi[-rv:]
                    prod = so * v[oi2]
                    if oi1:
                        gprimesum += as_tensor(prod, oi1)
                    else:
                        gprimesum += prod

        return gprimesum

    def coordinate_derivative(self, o):
        o = o.ufl_operands
        return CoordinateDerivative(map_expr_dag(self, o[0]), o[1], o[2], o[3])

    def base_form_operator(self, o, *dfs):
        """If d_coeff = 0 => BaseFormOperator's derivative is taken wrt a variable => we call the appropriate handler
        Otherwise => differentiation done wrt the BaseFormOperator (dF/dN[Nhat]) => we treat o as a Coefficient
        """
        d_coeff = self.coefficient(o)
        # It also handles the non-scalar case
        if d_coeff == 0:
            self.pending_operations += (o,)
        return d_coeff

    # -- Handlers for BaseForm objects -- #

    def cofunction(self, o):
        # Same rule than for Coefficient except that we use a Coargument.
        # The coargument is already attached to the class (self._v)
        # which `self.coefficient` relies on.
        dc = self.coefficient(o)
        if dc == 0:
            # Convert ufl.Zero into ZeroBaseForm
            return ZeroBaseForm(self._v)
        return dc

    def coargument(self, o):
        # Same rule than for Argument (da/dw == 0).
        dc = self.argument(o)
        if dc == 0:
            # Convert ufl.Zero into ZeroBaseForm
            return ZeroBaseForm(o.arguments() + self._v)
        return dc

    def matrix(self, M):
        # Matrix rule: D_w[v](M) = v if M == w else 0
        # We can't differentiate wrt a matrix so always return zero in the appropriate space
        return ZeroBaseForm(M.arguments() + self._v)


class BaseFormOperatorDerivativeRuleset(GateauxDerivativeRuleset):
    """Apply AFD (Automatic Functional Differentiation) to BaseFormOperator.

    Implements rules for the Gateaux derivative D_w[v](...) defined as

        D_w[v](B) = d/dtau B(w+tau v)|tau=0

    where B is a ufl.BaseFormOperator
    """

    def __init__(self, coefficients, arguments, coefficient_derivatives):
        GateauxDerivativeRuleset.__init__(self, coefficients, arguments, coefficient_derivatives)

    # --- BaseFormOperator handlers

    def interp(self, I, dw):
        # Interp rule: D_w[v](I(w, v*)) = I(v, v*), by linearity of Interp!
        if not dw:
            # I doesn't depend on w:
            #  -> It also covers the Hessian case since Interp is linear,
            #     e.g. D_w[v](D_w[v](I(w, v*))) = D_w[v](I(v, v*)) = 0 (since w not found).
            return ZeroBaseForm(I.arguments() + self._v)
        return I._ufl_expr_reconstruct_(expr=dw)


class DerivativeRuleDispatcher(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        # caches for reuse in the dispatched transformers
        self.vcaches = defaultdict(dict)
        self.rcaches = defaultdict(dict)

        # Record the operations delayed to the derivative expansion phase:
        # Example: dN(u)/du where `N` is a BaseFormOperator and `u` a Coefficient
        self.pending_operations = ()

    def terminal(self, o):
        return o

    def derivative(self, o):
        raise NotImplementedError(f"Missing derivative handler for {type(o).__name__}.")

    ufl_type = MultiFunction.reuse_if_untouched

    def grad(self, o, f):
        rules = GradRuleset(o.ufl_shape[-1])
        key = (GradRuleset, o.ufl_shape[-1])
        return map_expr_dag(rules, f,
                            vcache=self.vcaches[key],
                            rcache=self.rcaches[key])

    def reference_grad(self, o, f):
        rules = ReferenceGradRuleset(o.ufl_shape[-1])  # FIXME: Look over this and test better.
        key = (ReferenceGradRuleset, o.ufl_shape[-1])
        return map_expr_dag(rules, f,
                            vcache=self.vcaches[key],
                            rcache=self.rcaches[key])

    def variable_derivative(self, o, f, dummy_v):
        op = o.ufl_operands[1]
        rules = VariableRuleset(op)
        key = (VariableRuleset, op)
        return map_expr_dag(rules, f,
                            vcache=self.vcaches[key],
                            rcache=self.rcaches[key])

    def coefficient_derivative(self, o, f, dummy_w, dummy_v, dummy_cd):
        dummy, w, v, cd = o.ufl_operands
        rules = GateauxDerivativeRuleset(w, v, cd)
        key = (GateauxDerivativeRuleset, w, v, cd)
        mapped_expr = map_expr_dag(rules, f,
                                   vcache=self.vcaches[key],
                                   rcache=self.rcaches[key])
        # We need to go through the dag first to record the pending operations
        var, der_kwargs, *base_form_ops = rules.pending_operations
        # Need to account for pending operations that have been stored in other integrands
        base_form_ops = self.pending_operations[2:] + tuple(base_form_ops)
        self.pending_operations = (var, der_kwargs, *base_form_ops)
        return mapped_expr

    def base_form_operator_derivative(self, o, f, dummy_w, dummy_v, dummy_cd):
        dummy, w, v, cd = o.ufl_operands
        rules = BaseFormOperatorDerivativeRuleset(w, v, cd)
        key = (BaseFormOperatorDerivativeRuleset, w, v, cd)
        if isinstance(f, ZeroBaseForm):
            arg, = v.ufl_operands
            arguments = f.arguments()
            # derivative(F, u, du) with `du` a Coefficient
            # is equivalent to taking the action of the derivative.
            # In that case, we don't add arguments to `ZeroBaseForm`.
            if isinstance(arg, BaseArgument):
                arguments += (arg,)
            return ZeroBaseForm(arguments)
        mapped_expr = map_expr_dag(rules, f,
                                   vcache=self.vcaches[key],
                                   rcache=self.rcaches[key])
        mapped_f = rules.coefficient(f)
        if mapped_f != 0:
            # If dN/dN needs to return an Argument in N space
            # with N a BaseFormOperator.
            return mapped_f
        # We need to go through the dag first to record the pending operations
        var, der_kwargs, *base_form_ops = rules.pending_operations
        # Need to account for pending operations that have been stored in other integrands
        base_form_ops = self.pending_operations[2:] + tuple(base_form_ops)
        self.pending_operations = (var, der_kwargs, *base_form_ops)
        return mapped_expr

    def coordinate_derivative(self, o, f, dummy_w, dummy_v, dummy_cd):
        o_ = o.ufl_operands
        key = (CoordinateDerivative, o_[0])
        return CoordinateDerivative(map_expr_dag(self, o_[0],
                                                 vcache=self.vcaches[key],
                                                 rcache=self.rcaches[key]),
                                    o_[1], o_[2], o_[3])

    def indexed(self, o, Ap, ii):  # TODO: (Partially) duplicated in generic rules
        # Reuse if untouched
        if Ap is o.ufl_operands[0]:
            return o

        # Untangle as_tensor(C[kk], jj)[ii] -> C[ll] to simplify
        # resulting expression
        if isinstance(Ap, ComponentTensor):
            B, jj = Ap.ufl_operands
            if isinstance(B, Indexed):
                C, kk = B.ufl_operands

                kk = list(kk)
                if all(j in kk for j in jj):
                    rep = dict(zip(jj, ii))
                    Cind = [rep.get(k, k) for k in kk]
                    expr = Indexed(C, MultiIndex(tuple(Cind)))
                    assert expr.ufl_free_indices == o.ufl_free_indices
                    assert expr.ufl_shape == o.ufl_shape
                    return expr

        # Otherwise a more generic approach
        r = len(Ap.ufl_shape) - len(ii)
        if r:
            kk = indices(r)
            op = Indexed(Ap, MultiIndex(ii.indices() + kk))
            op = as_tensor(op, kk)
        else:
            op = Indexed(Ap, ii)
        return op


def apply_derivatives(expression):
    # Note that `expression` can be a Form, an Expr or a BaseFormOperator.
    # Notation: Let `var` be the thing we are differentating with respect to.

    rules = DerivativeRuleDispatcher()

    # If we hit an external operator, then if `var` is:
    #    - an ExternalOperator → Return d(expression)/dw where w is the coefficient produced by the external operator `var`.
    #    - else → Record the external operator on the MultiFunction object and returns 0.
    # Example:
    #    → If derivative(F(u, N(u); v), u) was taken the following line would compute `\frac{\partial F}{\partial u}`.
    dexpression_dvar = map_integrand_dags(rules, expression)

    # Get the recorded delayed operations
    pending_operations = rules.pending_operations

    if len(pending_operations) <= 2:
        return dexpression_dvar

    # Don't take into account empty Forms
    if not (isinstance(dexpression_dvar, Form) and len(dexpression_dvar.integrals()) == 0):
        dexpression_dvar = (dexpression_dvar,)
    else:
        dexpression_dvar = ()

    # Retrieve the base form operators, var, and the argument and coefficient_derivatives for `derivative`
    var, der_kwargs, *base_form_ops = pending_operations
    for N in sorted(set(base_form_ops), key=lambda x: x.count()):
        # Replace dexpr/dvar by dexpr/dN. We don't use `apply_derivatives` since
        # the differentiation is done via `\partial` and not `d`.
        dexpr_dN = map_integrand_dags(rules, replace_derivative_nodes(expression, {var.ufl_operands[0]: N}))
        # Add the BaseFormOperatorDerivative node
        # TODO: Should we use `derivative` here to take into account Extop/Interp node?
        dN_dvar = apply_derivatives(BaseFormOperatorDerivative(N, var, **der_kwargs))
        # Sum the Action: dF/du = \partial F/\partial u + \sum_{i=1,...} Action(dF/dNi, dNi/du)
        if not (isinstance(dexpr_dN, Form) and len(dexpr_dN.integrals()) == 0):
            # In this case: Action <=> ufl.action since `dN_var` has 2 arguments.
            # We use Action to handle the trivial case dN_dvar = 0.
            dexpression_dvar += (Action(dexpr_dN, dN_dvar),)
    return sum(dexpression_dvar)


class CoordinateDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AFD (Automatic Functional Differentiation) to expression.

    Implements rules for the Gateaux derivative D_w[v](...) defined as

        D_w[v](e) = d/dtau e(w+tau v)|tau=0

    where 'e' is a ufl form after pullback and w is a SpatialCoordinate.

    """

    def __init__(self, coefficients, arguments, coefficient_derivatives):
        GenericDerivativeRuleset.__init__(self, var_shape=())

        # Type checking
        if not isinstance(coefficients, ExprList):
            raise ValueError("Expecting a ExprList of coefficients.")
        if not isinstance(arguments, ExprList):
            raise ValueError("Expecting a ExprList of arguments.")
        if not isinstance(coefficient_derivatives, ExprMapping):
            raise ValueError("Expecting a coefficient-coefficient ExprMapping.")

        # The coefficient(s) to differentiate w.r.t. and the
        # argument(s) s.t. D_w[v](e) = d/dtau e(w+tau v)|tau=0
        self._w = coefficients.ufl_operands
        self._v = arguments.ufl_operands
        self._w2v = {w: v for w, v in zip(self._w, self._v)}

        # Build more convenient dict {f: df/dw} for each coefficient f
        # where df/dw is nonzero
        cd = coefficient_derivatives.ufl_operands
        self._cd = {cd[2 * i]: cd[2 * i + 1] for i in range(len(cd) // 2)}

    # Explicitly defining dg/dw == 0
    geometric_quantity = GenericDerivativeRuleset.independent_terminal

    # Explicitly defining da/dw == 0
    argument = GenericDerivativeRuleset.independent_terminal

    def coefficient(self, o):
        raise NotImplementedError("CoordinateDerivative of coefficient in physical space is not implemented.")

    def grad(self, o):
        raise NotImplementedError("CoordinateDerivative grad in physical space is not implemented.")

    def spatial_coordinate(self, o):
        do = self._w2v.get(o)
        # d x /d x => Argument(x.function_space())
        if do is not None:
            return do
        else:
            raise NotImplementedError("CoordinateDerivative found a SpatialCoordinate that is different from the one being differentiated.")

    def reference_value(self, o):
        do = self._cd.get(o)
        if do is not None:
            return do
        else:
            return self.independent_terminal(o)

    def reference_grad(self, g):
        # d (grad_X(...(x)) / dx => grad_X(...(Argument(x.function_space()))
        o = g
        ngrads = 0
        while isinstance(o, ReferenceGrad):
            o, = o.ufl_operands
            ngrads += 1
        if not (isinstance(o, SpatialCoordinate) or isinstance(o.ufl_operands[0], FormArgument)):
            raise ValueError(f"Expecting gradient of a FormArgument, not {ufl_err_str(o)}")

        def apply_grads(f):
            for i in range(ngrads):
                f = ReferenceGrad(f)
            return f

        # Find o among all w without any indexing, which makes this
        # easy
        for (w, v) in zip(self._w, self._v):
            if o == w and isinstance(v, ReferenceValue) and isinstance(v.ufl_operands[0], FormArgument):
                # Case: d/dt [w + t v]
                return apply_grads(v)
        return self.independent_terminal(o)

    def jacobian(self, o):
        # d (grad_X(x))/d x => grad_X(Argument(x.function_space())
        for (w, v) in zip(self._w, self._v):
            if extract_unique_domain(o) == extract_unique_domain(w) and isinstance(v.ufl_operands[0], FormArgument):
                return ReferenceGrad(v)
        return self.independent_terminal(o)


class CoordinateDerivativeRuleDispatcher(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        self.vcache = defaultdict(dict)
        self.rcache = defaultdict(dict)

    def terminal(self, o):
        return o

    def derivative(self, o):
        raise NotImplementedError(f"Missing derivative handler for {type(o).__name__}.")

    expr = MultiFunction.reuse_if_untouched

    def grad(self, o):
        return o

    def reference_grad(self, o):
        return o

    def coefficient_derivative(self, o):
        return o

    def coordinate_derivative(self, o, f, w, v, cd):
        from ufl.algorithms import extract_unique_elements
        for space in extract_unique_elements(o):
            if space.mapping() == "custom":
                raise NotImplementedError(
                    "CoordinateDerivative is not supported for elements with custom pull back.")

        _, w, v, cd = o.ufl_operands
        rules = CoordinateDerivativeRuleset(w, v, cd)
        key = (CoordinateDerivativeRuleset, w, v, cd)
        return map_expr_dag(rules, f, vcache=self.vcache[key], rcache=self.rcache[key])


def apply_coordinate_derivatives(expression):
    rules = CoordinateDerivativeRuleDispatcher()
    return map_integrand_dags(rules, expression)
