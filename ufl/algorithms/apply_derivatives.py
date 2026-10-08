"""Apply derivatives algorithm which computes the derivatives of a form of expression."""

# Copyright (C) 2008-2026 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

import warnings
from functools import singledispatchmethod
from math import pi

import numpy as np

from ufl.action import Action
from ufl.algorithms.analysis import (
    extract_arguments,
    extract_coefficients,
    extract_type,
)
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.remove_complex_nodes import remove_complex_nodes
from ufl.algorithms.replace import replace
from ufl.argument import Argument, BaseArgument, Coargument
from ufl.averaging import CellAvg, FacetAvg
from ufl.checks import is_cellwise_constant
from ufl.classes import (
    Abs,
    CellCoordinate,
    Coefficient,
    Cofunction,
    ComponentTensor,
    Conj,
    Constant,
    ConstantValue,
    Division,
    Expr,
    ExprList,
    ExprMapping,
    FacetNormal,
    FloatValue,
    FormArgument,
    GeometricQuantity,
    Grad,
    Identity,
    Imag,
    Indexed,
    IndexSum,
    Jacobian,
    JacobianDeterminant,
    JacobianInverse,
    Label,
    ListTensor,
    Power,
    Product,
    Real,
    ReferenceGrad,
    ReferenceValue,
    SpatialCoordinate,
    Sum,
    Variable,
    Zero,
)
from ufl.conditional import BinaryCondition, Conditional, NotCondition
from ufl.constantvalue import is_true_ufl_scalar, is_ufl_scalar
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.expr import ufl_err_str
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.core.multiindex import FixedIndex, MultiIndex, indices
from ufl.core.terminal import Terminal
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.corealg.traversal import unique_pre_traversal
from ufl.differentiation import (
    BaseFormCoordinateDerivative,
    BaseFormDerivative,
    CoefficientDerivative,
    CoordinateDerivative,
    Derivative,
    VariableDerivative,
)
from ufl.domain import MeshSequence, extract_unique_domain
from ufl.form import BaseForm, Form, FormSum, ZeroBaseForm
from ufl.mathfunctions import (
    Acos,
    Asin,
    Atan,
    Atan2,
    BesselI,
    BesselJ,
    BesselK,
    BesselY,
    Cos,
    Cosh,
    Erf,
    Exp,
    Ln,
    MathFunction,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
)
from ufl.matrix import Matrix
from ufl.operators import (
    MaxValue,
    MinValue,
    bessel_I,
    bessel_J,
    bessel_K,
    bessel_Y,
    cell_avg,
    conditional,
    cos,
    cosh,
    exp,
    facet_avg,
    ln,
    sign,
    sin,
    sinh,
    sqrt,
)
from ufl.pullback import CustomPullback, PhysicalPullback
from ufl.restriction import Restricted
from ufl.tensors import as_scalar, as_scalars, as_tensor, unit_indexed_tensor, unwrap_list_tensor

# TODO: Add more rulesets?
# - DivRuleset
# - CurlRuleset
# - ReferenceGradRuleset
# - ReferenceDivRuleset


def flatten_domain_element(domain, element):
    """Return the flattened (domain, element) pairs for mixed domain problems.

    Args:
        domain: `Mesh` or `MeshSequence`.
        element: `FiniteElement`.

    Returns:
        Nested tuples of (domain, element) pairs; just ((domain, element),)
        if domain is a `Mesh` (and not a `MeshSequence`).

    """
    if not isinstance(domain, MeshSequence):
        return ((domain, element),)
    flattened = ()
    assert len(domain) == len(element.sub_elements)
    for d, e in zip(domain, element.sub_elements):
        flattened += flatten_domain_element(d, e)
    return flattened


class GenericDerivativeRuleset(DAGTraverser):
    """A generic derivative."""

    def __init__(
        self,
        var_shape: tuple,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        self._var_shape = var_shape

    def unexpected(self, o):
        """Raise error about unexpected type."""
        raise ValueError(f"Unexpected type {o._ufl_class_.__name__} in AD rules.")

    def override(self, o):
        """Raise error about overriding."""
        raise ValueError(
            f"Type {o._ufl_class_.__name__} must be overridden in specialized AD rule set."
        )

    # --- Some types just don't have any derivative, this is just to
    # --- make algorithm structure generic

    def non_differentiable_terminal(self, o):
        """Return the non-differentiated object.

        Labels and indices are not differentiable: it's convenient to
        return the non-differentiated object.
        """
        return o

    # --- Helper functions for creating zeros with the right shapes

    def independent_terminal(self, o):
        """A zero with correct shape for terminals independent of diff. variable."""
        return Zero(o.ufl_shape + self._var_shape)

    def independent_operator(self, o):
        """A zero with correct shape and indices for operators independent of diff. variable."""
        return Zero(o.ufl_shape + self._var_shape, o.ufl_free_indices, o.ufl_index_dimensions)

    # --- Error checking for missing handlers and unexpected types

    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        """Raise error."""
        raise ValueError(
            f"Missing differentiation handler for type {o._ufl_class_.__name__}. "
            "Have you added a new type?"
        )

    @process.register(Derivative)
    def _(self, o: Expr) -> Expr:
        """Raise error."""
        raise ValueError(
            f"Unhandled derivative type {o._ufl_class_.__name__}, "
            "nested differentiation has failed."
        )

    @process.register(Label)
    @process.register(MultiIndex)
    def _(self, o: Expr) -> Expr:
        return self.non_differentiable_terminal(o)

    # --- All derivatives need to define grad and averaging

    @process.register(Grad)
    @process.register(CellAvg)
    @process.register(FacetAvg)
    def _(self, o: Expr) -> Expr:
        return self.override(o)

    # --- Default rules for terminals

    # Literals are by definition independent of any differentiation variable
    @process.register(ConstantValue)
    # Constants are independent of any differentiation
    @process.register(Constant)
    def _(self, o: Expr) -> Expr:
        return self.independent_terminal(o)

    # Zero may have free indices
    @process.register(Zero)
    def _(self, o: Expr) -> Expr:
        return self.independent_operator(o)

    # Rules for form arguments must be specified in specialized rule set
    @process.register(FormArgument)
    # Rules for geometric quantities must be specified in specialized rule set
    @process.register(GeometricQuantity)
    def _(self, o: Expr) -> Expr:
        return self.override(o)

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

    @process.register(Variable)
    def _(self, o: Expr) -> Expr:
        """Differentiate a variable."""
        op, _ = o.ufl_operands
        return self(op)

    # --- Indexing and component handling

    @process.register(Indexed)
    @DAGTraverser.postorder
    def _(self, o: Indexed, Ap: Expr, ii: MultiIndex) -> Expr:
        """Differentiate an indexed."""
        # Propagate zeros
        if isinstance(Ap, Zero):
            return self.independent_operator(o)
        r = len(Ap.ufl_shape) - len(ii)
        if r:
            kk = indices(r)
            op = Indexed(Ap, MultiIndex(ii.indices() + kk))
            op = as_tensor(op, kk)
        else:
            op = Indexed(Ap, ii)
        return op

    @process.register(ListTensor)
    def _(self, o: Expr) -> Expr:
        """Differentiate a list_tensor."""
        return ListTensor(*(self(op) for op in o.ufl_operands))

    @process.register(ComponentTensor)
    @DAGTraverser.postorder
    def _(self, o: ComponentTensor, Ap: Expr, ii: MultiIndex) -> Expr:
        """Differentiate a component_tensor."""
        if isinstance(Ap, Zero):
            op = self.independent_operator(o)
        else:
            Ap, jj = as_scalar(Ap)
            op = as_tensor(Ap, ii.indices() + jj)
        return op

    # --- Algebra operators

    @process.register(IndexSum)
    @DAGTraverser.postorder
    def _(self, o: Expr, Ap: Expr, ii: Expr) -> Expr:
        """Differentiate an index_sum."""
        return IndexSum(Ap, ii)

    @process.register(Sum)
    @DAGTraverser.postorder
    def _(self, o: Expr, da: Expr, db: Expr) -> Expr:
        """Differentiate a sum."""
        return da + db

    @process.register(Product)
    @DAGTraverser.postorder
    def _(self, o: Expr, da: Expr, db: Expr) -> Expr:
        """Differentiate a product."""
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

    @process.register(Division)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr, gp: Expr) -> Expr:
        """Differentiate a division."""
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

    @process.register(Power)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr, gp: Expr) -> Expr:
        """Differentiate a power."""
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
            op = fp * g * f ** (g - 1)
        else:
            # Note: This produces expressions like (1/w)*w**5 instead of w**4
            # op = o * (fp * g / f + gp * ln(f)) # This reuses o
            op = f ** (g - 1) * (
                g * fp + f * ln(f) * gp
            )  # This gives better accuracy in dolfin integration test
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

    @process.register(Abs)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr) -> Expr:
        """Differentiate an abs."""
        (f,) = o.ufl_operands
        # return conditional(eq(f, 0), 0, Product(sign(f), df)) abs is
        # not complex differentiable, so we workaround the case of a
        # real F in complex mode by defensively casting to real inside
        # the sign.
        return sign(Real(f)) * df

    # --- Complex algebra

    @process.register(Conj)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr) -> Expr:
        """Differentiate a conj."""
        return Conj(df)

    @process.register(Real)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr) -> Expr:
        """Differentiate a real."""
        return Real(df)

    @process.register(Imag)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr) -> Expr:
        """Differentiate a imag."""
        return Imag(df)

    # --- Mathfunctions

    @process.register(MathFunction)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr) -> Expr:
        """Differentiate a math_function."""
        # FIXME: Introduce a UserOperator type instead of this hack
        # and define user derivative() function properly
        if hasattr(o, "derivative"):
            return df * o.derivative()
        else:
            raise ValueError("Unknown math function.")

    @process.register(Sqrt)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a sqrt."""
        return fp / (2 * o)

    @process.register(Exp)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate an exp."""
        return fp * o

    @process.register(Ln)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a ln."""
        (f,) = o.ufl_operands
        if isinstance(f, Zero):
            raise ZeroDivisionError()
        return fp / f

    @process.register(Cos)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a cos."""
        (f,) = o.ufl_operands
        return fp * -sin(f)

    @process.register(Sin)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a sin."""
        (f,) = o.ufl_operands
        return fp * cos(f)

    @process.register(Tan)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a tan."""
        (f,) = o.ufl_operands
        return 2.0 * fp / (cos(2.0 * f) + 1.0)

    @process.register(Cosh)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a cosh."""
        (f,) = o.ufl_operands
        return fp * sinh(f)

    @process.register(Sinh)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a sinh."""
        (f,) = o.ufl_operands
        return fp * cosh(f)

    @process.register(Tanh)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a tanh."""
        (f,) = o.ufl_operands

        def sech(y):
            return (2.0 * cosh(y)) / (cosh(2.0 * y) + 1.0)

        return fp * sech(f) ** 2

    @process.register(Acos)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate an acos."""
        (f,) = o.ufl_operands
        return -fp / sqrt(1.0 - f**2)

    @process.register(Asin)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate an asin."""
        (f,) = o.ufl_operands
        return fp / sqrt(1.0 - f**2)

    @process.register(Atan)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate an atan."""
        (f,) = o.ufl_operands
        return fp / (1.0 + f**2)

    @process.register(Atan2)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr, gp: Expr) -> Expr:
        """Differentiate an atan2."""
        f, g = o.ufl_operands
        return (g * fp - f * gp) / (f**2 + g**2)

    @process.register(Erf)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate an erf."""
        (f,) = o.ufl_operands
        return fp * (2.0 / sqrt(pi) * exp(-(f**2)))

    # --- Bessel functions

    @process.register(BesselJ)
    @DAGTraverser.postorder
    def _(self, o: Expr, nup: Expr, fp: Expr) -> Expr:
        """Differentiate a bessel_j."""
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError(
                "Differentiation of bessel function w.r.t. nu is not supported."
            )

        if isinstance(nu, Zero):
            op = -bessel_J(1, f)
        else:
            op = 0.5 * (bessel_J(nu - 1, f) - bessel_J(nu + 1, f))
        return op * fp

    @process.register(BesselY)
    @DAGTraverser.postorder
    def _(self, o: Expr, nup: Expr, fp: Expr) -> Expr:
        """Differentiate a bessel_y."""
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError(
                "Differentiation of bessel function w.r.t. nu is not supported."
            )

        if isinstance(nu, Zero):
            op = -bessel_Y(1, f)
        else:
            op = 0.5 * (bessel_Y(nu - 1, f) - bessel_Y(nu + 1, f))
        return op * fp

    @process.register(BesselI)
    @DAGTraverser.postorder
    def _(self, o: Expr, nup: Expr, fp: Expr) -> Expr:
        """Differentiate a bessel_i."""
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError(
                "Differentiation of bessel function w.r.t. nu is not supported."
            )

        if isinstance(nu, Zero):
            op = bessel_I(1, f)
        else:
            op = 0.5 * (bessel_I(nu - 1, f) + bessel_I(nu + 1, f))
        return op * fp

    @process.register(BesselK)
    @DAGTraverser.postorder
    def _(self, o: Expr, nup: Expr, fp: Expr) -> Expr:
        """Differentiate a bessel_k."""
        nu, f = o.ufl_operands
        if not (nup is None or isinstance(nup, Zero)):
            raise NotImplementedError(
                "Differentiation of bessel function w.r.t. nu is not supported."
            )

        if isinstance(nu, Zero):
            op = -bessel_K(1, f)
        else:
            op = -0.5 * (bessel_K(nu - 1, f) + bessel_K(nu + 1, f))
        return op * fp

    # --- Restrictions

    @process.register(Restricted)
    @DAGTraverser.postorder
    def _(self, o: Restricted, fp: Expr) -> Expr:
        """Differentiate a restricted."""
        # Restriction and differentiation commutes
        if isinstance(fp, ConstantValue):
            return fp  # TODO: Add simplification to Restricted instead?
        else:
            return fp(o._side)  # (f+-)' == (f')+-

    # --- Conditionals

    @process.register(BinaryCondition)
    def _(self, o: BinaryCondition) -> Expr:
        """Differentiate a binary_condition."""
        raise RuntimeError("Can not differentiate a binary_condition.")

    @process.register(NotCondition)
    def _(self, o: NotCondition) -> Expr:
        """Differentiate a not_condition."""
        raise RuntimeError("Can not differentiate a not_condition.")

    @process.register(Conditional)
    @DAGTraverser.postorder_only_children([1, 2])
    def _(self, o: Expr, dt: Expr, df: Expr) -> Expr:
        """Differentiate a conditional."""
        if isinstance(dt, Zero) and isinstance(df, Zero):
            # Assuming dt and df have the same indices here, which
            # should be the case
            return dt
        else:
            # Not placing t[1],f[1] outside, allowing arguments inside
            # conditionals.  This will make legacy ffc fail, but
            # should work with uflacs.
            c = o.ufl_operands[0]
            return conditional(c, dt, df)

    @process.register(MaxValue)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr, dg: Expr) -> Expr:
        """Differentiate a max_value."""
        # d/dx max(f, g) =
        # f > g: df/dx
        # f < g: dg/dx
        # Placing df,dg outside here to avoid getting arguments inside
        # conditionals
        f, g = o.ufl_operands
        dc = conditional(f > g, 1, 0)
        return dc * df + (1.0 - dc) * dg

    @process.register(MinValue)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr, dg: Expr) -> Expr:
        """Differentiate a min_value."""
        # d/dx min(f, g) =
        #  f < g: df/dx
        #  else: dg/dx
        #  Placing df,dg outside here to avoid getting arguments
        #  inside conditionals
        f, g = o.ufl_operands
        dc = conditional(f < g, 1, 0)
        return dc * df + (1.0 - dc) * dg


class GradRuleset(GenericDerivativeRuleset):
    """Take the grad derivative."""

    def __init__(
        self,
        geometric_dimension: int,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(
            (geometric_dimension,),
            compress=compress,
            visited_cache=visited_cache,
            result_cache=result_cache,
        )
        self._Id = Identity(geometric_dimension)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    # --- Specialized rules for geometric quantities

    @process.register(GeometricQuantity)
    def _(self, o: Expr) -> Expr:
        """Differentiate a geometric_quantity.

        Default for geometric quantities is do/dx = 0 if piecewise constant,
        otherwise transform derivatives to reference derivatives.
        Override for specific types if other behaviour is needed.
        """
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        else:
            domain = extract_unique_domain(o)
            K = JacobianInverse(domain)
            Do = grad_to_reference_grad(o, K)
            return Do

    @process.register(JacobianInverse)
    def _(self, o: JacobianInverse) -> Expr:
        """Differentiate a jacobian_inverse."""
        # grad(K) == K_ji rgrad(K)_rj
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        if not o._ufl_is_terminal_:
            raise ValueError("ReferenceValue can only wrap a terminal")
        Do = grad_to_reference_grad(o, o)
        return Do

    # TODO: Add more explicit geometry type handlers here, with
    # non-affine domains several should be non-zero.

    @process.register(SpatialCoordinate)
    def _(self, o: Expr) -> Expr:
        """Differentiate a spatial_coordinate.

        dx/dx = I.
        """
        return self._Id

    @process.register(CellCoordinate)
    def _(self, o: Expr) -> Expr:
        """Differentiate a cell_coordinate.

        dX/dx = inv(dx/dX) = inv(J) = K.
        """
        # FIXME: Is this true for manifolds? What about orientation?
        return JacobianInverse(extract_unique_domain(o))

    # --- Specialized rules for form arguments

    @process.register(BaseFormOperator)
    def _(self, o: Expr) -> Expr:
        """Differentiate a base_form_operator."""
        # Push the grad through the operator is not legal in most cases:
        #    -> Not enouth regularity for chain rule to hold!
        # By the time we evaluate `grad(o)`, the operator `o` will have
        # been assembled and substituted by its output.
        return Grad(o)

    @process.register(Coefficient)
    def _(self, o: Expr) -> Expr:
        """Differentiate a coefficient."""
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        return Grad(o)

    @process.register(Argument)
    def _(self, o: Expr) -> Expr:
        """Differentiate an argument."""
        # TODO: Enable this after fixing issue#13, unless we move
        # simplificat ion to a separate stage?
        # if is_cellwise_constant(o):
        #     # Collapse gradient of cellwise constant function to zero
        #     # TODO: Missing this type
        #     return AnnotatedZero(o.ufl_shape + self._var_shape, arguments=(o,))
        return Grad(o)

    # --- Rules for values or derivatives in reference frame

    @process.register(ReferenceValue)
    def _(self, o: ReferenceValue) -> Expr:
        """Differentiate a reference_value."""
        # grad(o) == grad(rv(f)) -> K_ji*rgrad(rv(f))_rj
        f = o.ufl_operands[0]
        if not (f._ufl_is_terminal_ or isinstance(f, Interpolate)):
            raise ValueError("ReferenceValue can only wrap a terminal")
        domain = extract_unique_domain(f, expand_mesh_sequence=False)
        if isinstance(domain, MeshSequence):
            element = f.ufl_function_space().ufl_element()  # type: ignore
            if element.num_sub_elements != len(domain):
                raise RuntimeError(f"{element.num_sub_elements} != {len(domain)}")
            # Get monolithic representation of rgrad(o); o might live in a mixed space.
            rgrad = ReferenceGrad(o)
            ref_dim = rgrad.ufl_shape[-1]
            # Apply K_ji(d) to the corresponding components of rgrad, store them in a list,
            # and put them back together at the end using as_tensor().
            components = []
            dofoffset = 0
            for d, e in flatten_domain_element(domain, element):
                esh = e.reference_value_shape
                ndof = int(np.prod(esh))
                assert ndof > 0
                if isinstance(e.pullback, PhysicalPullback):
                    if ref_dim != self._var_shape[0]:
                        raise NotImplementedError("""
                            PhysicalPullback not handled for immersed domain :
                            reference dim ({ref_dim}) != physical dim (self._var_shape[0])""")
                    for idx in range(ndof):
                        for i in range(ref_dim):
                            components.append(rgrad[(dofoffset + idx,) + (i,)])
                else:
                    K = JacobianInverse(d)
                    rdim, gdim = K.ufl_shape
                    if rdim > ref_dim:
                        # Note that rdim < ref_dim if mixed-dimensional.
                        raise RuntimeError(f"{rdim} > {ref_dim}")
                    if gdim != self._var_shape[0]:
                        raise RuntimeError(f"{gdim} != {self._var_shape[0]}")
                    # Note that rgrad[dofoffset + [0,ndof), [0,rdim)] are the components
                    # corresponding to (d, e).
                    # For each row, rgrad[dofoffset + idx, [0,rdim)], we apply
                    # K_ji(d)[[0,rdim), [0,gdim)].
                    for idx in range(ndof):
                        for i in range(gdim):
                            temp = Zero()
                            for j in range(rdim):
                                temp += rgrad[(dofoffset + idx,) + (j,)] * K[j, i]
                            components.append(temp)
                dofoffset += ndof
            return as_tensor(np.asarray(components).reshape(rgrad.ufl_shape[:-1] + self._var_shape))
        else:
            if isinstance(f.ufl_element().pullback, PhysicalPullback):  # type: ignore
                # TODO: Do we need to be more careful for immersed things?
                return ReferenceGrad(o)
            else:
                K = JacobianInverse(domain)
                return grad_to_reference_grad(o, K)

    @process.register(ReferenceGrad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_grad."""
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        # grad(o) == grad(rgrad(rv(f))) -> K_ji*rgrad(rgrad(rv(f)))_rj
        f = o.ufl_operands[0]
        valid_operand = f._ufl_is_in_reference_frame_ or isinstance(
            f, JacobianInverse | SpatialCoordinate | Jacobian | JacobianDeterminant | FacetNormal
        )
        if not valid_operand:
            raise ValueError("ReferenceGrad can only wrap a reference frame type!")
        domain = extract_unique_domain(f, expand_mesh_sequence=False)
        if isinstance(domain, MeshSequence):
            if not f._ufl_is_in_reference_frame_:
                raise RuntimeError("Expecting a reference frame type")
            while not f._ufl_is_terminal_:
                (f,) = f.ufl_operands
            element = f.ufl_function_space().ufl_element()
            if element.num_sub_elements != len(domain):
                raise RuntimeError(f"{element.num_sub_elements} != {len(domain)}")
            # Get monolithic representation of rgrad(o); o might live in a mixed space.
            rgrad = ReferenceGrad(o)
            ref_dim = rgrad.ufl_shape[-1]
            # Apply K_ji(d) to the corresponding components of rgrad, store them in a list,
            # and put them back together at the end using as_tensor().
            components = []
            dofoffset = 0
            for d, e in flatten_domain_element(domain, element):
                esh = e.reference_value_shape
                ndof = int(np.prod(esh))
                assert ndof > 0
                K = JacobianInverse(d)
                rdim, gdim = K.ufl_shape
                if rdim > ref_dim:
                    # Note that rdim < ref_dim if mixed-dimensional.
                    raise RuntimeError(f"{rdim} > {ref_dim}")
                if gdim != self._var_shape[0]:
                    raise RuntimeError(f"{gdim} != {self._var_shape[0]}")
                # Note that rgrad[dofoffset + [0,ndof), [0,rdim), [0,rdim)] are the components
                # corresponding to (d, e).
                # For each row, rgrad[dofoffset + idx, [0,rdim), [0,rdim)], we apply
                # K_ji(d)[[0,rdim), [0,gdim)].
                for idx in range(ndof):
                    for midx in np.ndindex(rgrad.ufl_shape[1:-1]):
                        for i in range(gdim):
                            temp = Zero()
                            for j in range(rdim):
                                temp += rgrad[(dofoffset + idx,) + midx + (j,)] * K[j, i]
                            components.append(temp)
                dofoffset += ndof
            if rgrad.ufl_shape[0] != dofoffset:
                raise RuntimeError(f"{rgrad.ufl_shape[0]} != {dofoffset}")
            return as_tensor(np.asarray(components).reshape(rgrad.ufl_shape[:-1] + self._var_shape))
        else:
            K = JacobianInverse(domain)
            return grad_to_reference_grad(o, K)

    # --- Nesting of gradients

    @process.register(Grad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a grad.

        Represent grad(grad(f)) as Grad(Grad(f)).
        """
        # Check that o is a "differential terminal"
        if not isinstance(o.ufl_operands[0], Grad | Terminal):
            raise ValueError("Expecting only grads applied to a terminal.")
        return Grad(o)

    def _grad(self, o):
        """Differentiate a _grad."""
        pass
        # TODO: Not sure how to detect that gradient of f is cellwise constant.
        #       Can we trust element degrees?
        # if is_cellwise_constant(o):
        #     return self.terminal(o)
        # TODO: Maybe we can ask "f.has_derivatives_of_order(n)" to check
        #       if we should make a zero here?
        # 1) n = count number of Grads, get f
        # 2) if not f.has_derivatives(n): return zero(...)

    @process.register(CellAvg)
    @process.register(FacetAvg)
    def _(self, o: Expr) -> Expr:
        return self.independent_operator(o)


def grad_to_reference_grad(o, K):
    """Relates grad(o) to reference_grad(o) using the Jacobian inverse.

    Args:
        o: Operand
        K: Jacobian inverse
    Returns:
        grad(o) written in terms of reference_grad(o) and K
    """
    r = indices(len(o.ufl_shape))
    i, j = indices(2)
    # grad(o) == K_ji rgrad(o)_rj
    Do = as_tensor(K[j, i] * ReferenceGrad(o)[r + (j,)], r + (i,))
    return Do


class ReferenceGradRuleset(GenericDerivativeRuleset):
    """Apply the reference grad derivative."""

    def __init__(
        self,
        topological_dimension: int,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(
            (topological_dimension,),
            compress=compress,
            visited_cache=visited_cache,
            result_cache=result_cache,
        )
        self._Id = Identity(topological_dimension)

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    # --- Specialized rules for geometric quantities

    @process.register(GeometricQuantity)
    def _(self, o: Expr) -> Expr:
        """Differentiate a geometric_quantity.

        dg/dX = 0 if piecewise constant, otherwise ReferenceGrad(g).
        """
        if is_cellwise_constant(o):
            return self.independent_terminal(o)
        else:
            # TODO: Which types does this involve? I don't think the
            # form compilers will handle this.
            return ReferenceGrad(o)

    @process.register(SpatialCoordinate)
    def _(self, o: Expr) -> Expr:
        """Differentiate a spatial_coordinate.

        dx/dX = J.
        """
        # Don't convert back to J, otherwise we get in a loop
        return ReferenceGrad(o)

    @process.register(CellCoordinate)
    def _(self, o: Expr) -> Expr:
        """Differentiate a cell_coordinate.

        dX/dX = I.
        """
        return self._Id

    # TODO: Add more geometry types here, with non-affine domains
    # several should be non-zero.

    # --- Specialized rules for form arguments

    @process.register(ReferenceValue)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_value."""
        if not (o.ufl_operands[0]._ufl_is_terminal_ or isinstance(o.ufl_operands[0], Interpolate)):
            raise ValueError("ReferenceValue can only wrap a terminal")
        return ReferenceGrad(o)

    @process.register(Coefficient)
    def _(self, o: Expr) -> Expr:
        """Differentiate a coefficient."""
        raise ValueError("Coefficient should be wrapped in ReferenceValue by now")

    @process.register(Argument)
    def _(self, o: Expr) -> Expr:
        """Differentiate an argument."""
        raise ValueError("Argument should be wrapped in ReferenceValue by now")

    # --- Nesting of gradients

    @process.register(Grad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a grad."""
        raise ValueError(
            f"Grad should have been transformed by this point, but got {type(o).__name__}."
        )

    @process.register(ReferenceGrad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_grad.

        Represent ref_grad(ref_grad(f)) as RefGrad(RefGrad(f)).
        """
        # Check that o is a "differential terminal"
        if not isinstance(o.ufl_operands[0], ReferenceGrad | ReferenceValue | Terminal):
            raise ValueError("Expecting only grads applied to a terminal.")
        return ReferenceGrad(o)

    @process.register(CellAvg)
    @process.register(FacetAvg)
    def _(self, o: Expr) -> Expr:
        return self.independent_operator(o)


class VariableRuleset(GenericDerivativeRuleset):
    """Differentiate with respect to a variable."""

    def __init__(
        self,
        var: Expr,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(
            var.ufl_shape, compress=compress, visited_cache=visited_cache, result_cache=result_cache
        )
        if var.ufl_free_indices:
            raise ValueError("Differentiation variable cannot have free indices.")
        self._variable = var
        self._Id = self._make_identity(self._var_shape)

    def _make_identity(self, sh):
        """Differentiate a _make_identity.

        Creates a higher order identity tensor to represent dv/dv.
        """
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

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(GeometricQuantity)
    def _(self, o: Expr) -> Expr:
        # Explicitly defining dg/dw == 0
        return self.independent_terminal(o)

    @process.register(Argument)
    def _(self, o: Expr) -> Expr:
        # Explicitly defining da/dw == 0
        return self.independent_terminal(o)

    @process.register(Coefficient)
    def _(self, o: Expr) -> Expr:
        """Differentiate a coefficient.

        df/dv = Id if v is f else 0.

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

    @process.register(Variable)
    @DAGTraverser.postorder
    def _(self, o: Expr, df: Expr, a: Expr) -> Expr:
        """Differentiate a variable."""
        v = self._variable
        if isinstance(v, Variable) and v.label() == a:
            # dv/dv = identity of rank 2*rank(v)
            return self._Id
        else:
            # df/v = df
            return df

    @process.register(Grad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a grad.

        Variable derivative of a gradient of a terminal must be 0.
        """
        # Check that o is a "differential terminal"
        if not isinstance(o.ufl_operands[0], Grad | Terminal):
            raise ValueError("Expecting only grads applied to a terminal.")
        return self.independent_terminal(o)

    # --- Rules for values or derivatives in reference frame

    @process.register(ReferenceValue)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_value."""
        # d/dv(o) == d/dv(rv(f)) = 0 if v is not f, or rv(dv/df)
        v = self._variable
        if isinstance(v, Coefficient) and o.ufl_operands[0] == v:
            if not v.ufl_element().pullback.is_identity:
                # FIXME: This is a bit tricky, instead of Identity it is
                #   actually inverse(transform), or we should rather not
                #   convert to reference frame in the first place
                raise ValueError(
                    "Missing implementation: To handle derivatives of rv(f) w.r.t. f for "
                    "mapped elements, rewriting to reference frame should not happen first..."
                )
            # dv/dv = identity of rank 2*rank(v)
            return self._Id
        else:
            # df/v = 0
            return self.independent_terminal(o)

    @process.register(ReferenceGrad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_grad.

        Variable derivative of a gradient of a terminal must be 0.
        """
        return self.independent_terminal(o)

    @process.register(CellAvg)
    @process.register(FacetAvg)
    def _(self, o: Expr) -> Expr:
        return self.independent_operator(o)


class GateauxDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AFD (Automatic Functional Differentiation) to expression.

    Implements rules for the Gateaux derivative D_w[v](...) defined as
    D_w[v](e) = d/dtau e(w+tau v)|tau=0.
    """

    def __init__(
        self,
        coefficients: ExprList,
        arguments: ExprList,
        coefficient_derivatives: ExprMapping,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(
            (), compress=compress, visited_cache=visited_cache, result_cache=result_cache
        )
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
        # The arguments that the derivative adds to a BaseForm. A Coefficient
        # direction, as in a tangent linear model, adds none.
        self._direction_arguments = tuple(
            a
            for v in self._v
            for a in ((v,) if isinstance(v, BaseArgument) else extract_arguments(v))
        )
        # Build more convenient dict {f: df/dw} for each coefficient f
        # where df/dw is nonzero
        cd = coefficient_derivatives.ufl_operands
        self._cd = {cd[2 * i]: cd[2 * i + 1] for i in range(len(cd) // 2)}

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    # --- Specialized rules for geometric quantities

    @process.register(GeometricQuantity)
    def _(self, o: Expr) -> Expr:
        # Explicitly defining dg/dw == 0
        return self.independent_terminal(o)

    @process.register(CellAvg)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a cell_avg."""
        # Cell average of a single function and differentiation
        # commutes, D_f[v](cell_avg(f)) = cell_avg(v)
        return cell_avg(fp)

    @process.register(FacetAvg)
    @DAGTraverser.postorder
    def _(self, o: Expr, fp: Expr) -> Expr:
        """Differentiate a facet_avg."""
        # Facet average of a single function and differentiation
        # commutes, D_f[v](facet_avg(f)) = facet_avg(v)
        return facet_avg(fp)

    @process.register(Argument)
    def _(self, o: Argument) -> Expr:
        # Explicitly defining da/dw == 0
        return self._process_argument(o)

    def _process_argument(self, o: Argument | Coargument) -> Zero:
        return self.independent_terminal(o)

    @process.register(Coefficient)
    def _(self, o: Coefficient) -> Expr:
        return self._process_coefficient(o)

    def _process_coefficient(self, o: Expr) -> Expr:
        """Differentiate an Expr or a BaseForm."""
        # Define dw/dw := d/ds [w + s v] = v

        # Return corresponding argument if we can find o among w
        do = self._w2v.get(o)  # type: ignore
        if do is not None:
            return do

        # Look for o among coefficient derivatives
        dos: Expr | tuple[Expr, ...] | None = self._cd.get(o)  # type: ignore
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
                raise ValueError(
                    "Got a tuple of arguments, expecting a "
                    "matching tuple of coefficient derivatives."
                )
            dosum = Zero(o.ufl_shape)
            for do, v in zip(dos, self._v):
                so, oi = as_scalar(do)
                rv = len(oi) - len(v.ufl_shape)
                oi1 = oi[:rv]
                oi2 = oi[rv:]
                prod = so * v[oi2]
                if oi1:
                    dosum += as_tensor(prod, oi1)
                else:
                    dosum += prod
            return dosum

    @process.register(ReferenceValue)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_value."""
        raise NotImplementedError(
            "Currently no support for ReferenceValue in CoefficientDerivative."
        )
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

    @process.register(ReferenceGrad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_grad."""
        if len(extract_coefficients(o)) > 0:
            raise NotImplementedError(
                "Currently no support for ReferenceGrad in CoefficientDerivative."
            )
        else:
            return Zero(o.ufl_shape)
        # TODO: This is implementable for regular derivative(M(f),f,v)
        #       but too messy if customized coefficient derivative
        #       relations are given by the user.  We would only need
        #       this to allow the user to write
        #       derivative(...ReferenceValue...,...).

    @process.register(Grad)
    def _(self, g: Expr) -> Expr:
        """Differentiate a grad."""
        # If we hit this type, it has already been propagated to a
        # coefficient (or grad of a coefficient) or a base form operator, # FIXME: Assert
        # this!  so we need to take the gradient of the variation or
        # return zero.  Complications occur when dealing with
        # derivatives w.r.t. single components...

        # Figure out how many gradients are around the inner terminal
        ngrads = 0
        o = g
        while isinstance(o, Grad):
            (o,) = o.ufl_operands
            ngrads += 1
        # `grad(N)` where N is a BaseFormOperator is treated as if `N` was a Coefficient.
        if not isinstance(o, FormArgument | BaseFormOperator):
            raise ValueError(f"Expecting gradient of a FormArgument, not {ufl_err_str(o)}.")

        def apply_grads(f):
            for i in range(ngrads):
                f = Grad(f)
            return f

        if isinstance(o, BaseFormOperator) and o not in self._w:
            # D[grad(N)] = grad(D[N])
            do = self(o)
            grad_ruleset = GradRuleset(g.ufl_shape[-1])
            for i in range(ngrads):
                do = grad_ruleset(do)
            return do

        # Find o among all w without any indexing, which makes this
        # easy
        for w, v in zip(self._w, self._v):
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
        for w, v in zip(self._w, self._v):
            wval: Expr
            # -- Analyse differentiation variable coefficient -- #

            # Can differentiate a Form wrt a BaseFormOperator
            if isinstance(w, FormArgument | BaseFormOperator):
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
                            gprimesum = gprimesum + compute_gprimeterm(
                                ngrads, vval, vcomp, wshape, wcomp
                            )
                elif isinstance(v, Zero):
                    pass

                else:
                    if wshape != ():
                        raise ValueError("Expecting scalar coefficient in this branch.")
                    # Case: d/dt [w + t v[...]]
                    wval, wcomp = w, ()

                    vval, vcomp = analyse_variation_argument(v)
                    gprimesum = gprimesum + compute_gprimeterm(ngrads, vval, vcomp, wshape, wcomp)

            elif isinstance(
                w, Indexed
            ):  # This path is tested in unit tests, but not actually used?
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
                    warnings.warn(f"Assuming d{{{0}}}/d{{{self._w}}} = 0.")
            else:
                # Make sure we have a tuple to match the self._v tuple
                if not isinstance(oprimes, tuple):
                    oprimes = (oprimes,)
                    if len(oprimes) != len(self._v):
                        raise ValueError(
                            "Got a tuple of arguments, expecting a"
                            " matching tuple of coefficient derivatives."
                        )

                # Compute dg/dw_j = dg/dw_h : v.
                # Since we may actually have a tuple of oprimes and vs
                # in a 'mixed' space, sum over them all to get the
                # complete inner product. Using indices to define a
                # non-compound inner product.
                for oprime, v in zip(oprimes, self._v):
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

    @process.register(CoordinateDerivative)
    @DAGTraverser.postorder_only_children([0])
    def _(self, o: Expr, o0: Expr) -> Expr:
        """Differentiate a coordinate_derivative."""
        _, o1, o2, o3 = o.ufl_operands
        return CoordinateDerivative(o0, o1, o2, o3)

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> Expr:
        """Differentiate a base_form_operator."""
        return self._process_coefficient(o)

    @process.register(Interpolate)
    @DAGTraverser.postorder
    def _(self, i_op: Interpolate, dw: Expr) -> Expr:
        """Differentiate an interpolate."""
        di_op = self._process_coefficient(i_op)
        if di_op != 0 or dw == 0:
            return di_op
        # Interpolate rule: D_w[v](i_op(w, v*)) = i_op(v, v*), by linearity of Interpolate!
        return i_op._ufl_expr_reconstruct_(dw)

    @process.register(ExternalOperator)
    @DAGTraverser.postorder
    def _(self, N: ExternalOperator, *dfs) -> Expr:
        """Differentiate an external_operator."""
        dN = self._process_coefficient(N)
        if dN != 0:
            return dN
        for i, df in enumerate(dfs):
            if df == 0:
                continue
            if len(extract_arguments(df)) == 0:
                raise NotImplementedError(
                    "Frechet derivative of external operators need to be provided!"
                )
            # dNdOi(..., Oi, ...; DOi(u)[v], ..., v*) is the Gateaux derivative of
            # N(..., Oi, ...; ..., v*) with respect to its i-th operand Oi in the
            # direction DOi(u)[v], e.g. N(u) = u**2 gives dNdu(u; uhat, v*) = 2 * u * uhat.
            derivatives = tuple(dj + int(i == j) for j, dj in enumerate(N.derivatives))
            dN += N._ufl_expr_reconstruct_(
                *N.ufl_operands,
                derivatives=derivatives,
                argument_slots=N.argument_slots() + (df,),
            )
        # N is linear in its argument slots, as in N(u; DM(u)[v0], v*) for N(M(u)).
        vstar, *slots = N.argument_slots()
        for i, slot in enumerate(slots):
            dslot = self(slot)
            if dslot == 0:
                continue
            dslots = (*slots[:i], dslot, *slots[i + 1 :])
            dN += N._ufl_expr_reconstruct_(*N.ufl_operands, argument_slots=(vstar, *dslots))
        return dN

    # -- Handlers for BaseForm objects -- #

    @process.register(Cofunction)
    def _(self, o: Cofunction) -> Expr:
        """Differentiate a cofunction."""
        # Same rule than for Coefficient except that we use a Coargument.
        # The coargument is already attached to the class (self._v)
        # which `self.coefficient` relies on.
        dc = self._process_coefficient(o)  # type: ignore
        if dc == 0:
            # Convert ufl.Zero into ZeroBaseForm
            return ZeroBaseForm(o.arguments() + self._direction_arguments)  # type: ignore
        return dc

    @process.register(Coargument)
    def _(self, o: Coargument) -> Expr:
        """Differentiate a coargument."""
        # Same rule than for Argument (da/dw == 0).
        dc = self._process_argument(o)
        if dc == 0:
            # Convert ufl.Zero into ZeroBaseForm
            return ZeroBaseForm(o.arguments() + self._direction_arguments)  # type: ignore
        return dc

    @process.register(Matrix)  # type: ignore
    def _(self, M: Matrix) -> BaseForm:
        """Differentiate a matrix."""
        # Matrix rule: D_w[v](M) = v if M == w else 0
        # We can't differentiate wrt a matrix so always return zero in
        # the appropriate space
        return ZeroBaseForm(M.arguments() + self._direction_arguments)

    @process.register(ZeroBaseForm)  # type: ignore
    def _(self, o: BaseForm) -> BaseForm:
        """Differentiate a zero_base_form."""
        # ZeroBaseForm is idempotent under differentiation: it stays zero,
        # gaining the new derivative direction as an extra argument.
        return ZeroBaseForm(o.arguments() + self._direction_arguments)


class BaseFormDerivativeRuleset(GateauxDerivativeRuleset):
    """Apply AFD (Automatic Functional Differentiation) to BaseForm.

    Implements rules for the Gateaux derivative D_w[v](B) where B is a
    BaseForm whose own derivatives have been expanded. The expressions in B,
    such as integrands, are differentiated with the GateauxDerivativeRuleset.
    """

    def __init__(
        self,
        coefficients: ExprList,
        arguments: ExprList,
        coefficient_derivatives: ExprMapping,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(
            coefficients,
            arguments,
            coefficient_derivatives,
            compress=compress,
            visited_cache=visited_cache,
            result_cache=result_cache,
        )
        self._expression_rules = GateauxDerivativeRuleset(
            coefficients, arguments, coefficient_derivatives
        )

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        """Differentiate an expression in a slot of a BaseForm."""
        return self._expression_rules(o)

    @process.register(Form)
    def _(self, o: Form) -> BaseForm:
        """Differentiate a form."""
        dform = map_integrands(self._expression_rules, o)
        if isinstance(dform, Form) and dform.empty():
            return ZeroBaseForm(o.arguments() + self._direction_arguments)
        return dform

    @process.register(FormSum)
    def _(self, o: FormSum) -> BaseForm:
        """Differentiate a form sum."""
        return FormSum(*((self(c), w) for c, w in zip(o.components(), o.weights())))

    @process.register(Action)
    def _(self, o: Action) -> BaseForm:
        """Differentiate an action with the Leibniz rule."""
        left, right = o.ufl_operands
        assert isinstance(left, BaseForm)
        if len(left.arguments()) != 1:
            raise NotImplementedError(
                "Action derivative not supported when the left argument is not a 1-form."
            )
        # Number the argument of `left` after the direction, so that the Action
        # contracts it and not the argument of the direction.
        (vleft,) = left.arguments()
        number = 1 + max((vleft.number(), *(a.number() for a in self._direction_arguments)))
        left_after_v = replace(left, {vleft: vleft.reconstruct(number=number)})
        return Action(self(left_after_v), right) + Action(left, self(right))

    @process.register(BaseFormOperator)
    def _(self, N: BaseFormOperator) -> Expr | BaseForm:
        """Differentiate a base form operator.

        N(u; v*) is linear in its dual slot v*, so the product rule gives
        D[N(u; v*)] = DN(u; v*) + N(u; Dv*), where N(u; Dv*) is the action
        of N(u; vhat) on Dv*, with vhat a new coargument in the dual slot.
        """
        dN_dN = self._process_coefficient(N)
        if dN_dN != 0:
            # Differentiation with respect to N itself.
            return dN_dN
        dN = self._expression_rules(N)

        vstar, *slots = N.argument_slots()
        # The Action contracts the last argument of N, so number vhat after the others.
        number = 1 + max(
            (a.number() for slot in slots for a in extract_type(slot, Argument, True)),
            default=-1,
        )
        primal_argument, *_ = vstar.arguments()
        vhat = primal_argument.reconstruct(
            function_space=primal_argument.ufl_function_space().dual(), number=number
        )
        Nhat = N._ufl_expr_reconstruct_(*N.ufl_operands, argument_slots=(vhat, *slots))
        return dN + Action(Nhat, self(vstar))


class DerivativeRuleDispatcher(DAGTraverser):
    """Dispatch a derivative rule."""

    def __init__(
        self,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        # Create DAGTraverser caches.
        self._dag_traverser_cache: dict[
            tuple[type, Expr | BaseForm]
            | tuple[type, Expr | BaseForm, Expr | BaseForm, Expr | BaseForm]
            | tuple[type, Expr | BaseForm, Expr | BaseForm, Expr | BaseForm, Expr | BaseForm],
            DAGTraverser,
        ] = {}

    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(Expr)
    @process.register(BaseForm)  # type: ignore
    def _(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Apply to expr and base form."""
        return self.reuse_if_untouched(o)

    @process.register(Terminal)
    def _(self, o: Terminal) -> Terminal:
        """Apply to a terminal."""
        return o

    @process.register(Form)
    def _(self, o: Form) -> Form:
        """Apply to a form in an operand of a base form."""
        # The integrals of a Form are not operands.
        return map_integrands(self, apply_algebra_lowering(o))

    @process.register(Derivative)
    def _(self, o: Derivative) -> Expr:
        """Apply to a derivative."""
        raise NotImplementedError(f"Missing derivative handler for {type(o).__name__}.")

    @process.register(Grad)
    @DAGTraverser.postorder
    def _(self, o: Grad, f: Expr | BaseForm) -> Expr:
        """Apply to a grad."""
        gdim = o.ufl_shape[-1]
        key = (GradRuleset, gdim)
        dag_traverser = self._dag_traverser_cache.setdefault(key, GradRuleset(gdim))
        return dag_traverser(f)  # type: ignore

    @process.register(ReferenceGrad)
    @DAGTraverser.postorder
    def _(self, o: ReferenceGrad, f: Expr | BaseForm) -> Expr | BaseForm:
        """Apply to a reference_grad."""
        tdim = o.ufl_shape[-1]
        key = (ReferenceGradRuleset, tdim)
        dag_traverser = self._dag_traverser_cache.setdefault(key, ReferenceGradRuleset(tdim))
        return dag_traverser(f)  # type: ignore

    @process.register(VariableDerivative)
    @DAGTraverser.postorder_only_children([0])
    def _(self, o: Expr, f: Expr | BaseForm) -> Expr | BaseForm:
        """Apply to a variable_derivative."""
        _, op = o.ufl_operands
        key = (VariableRuleset, op)
        dag_traverser = self._dag_traverser_cache.setdefault(key, VariableRuleset(op))
        return dag_traverser(f)  # type: ignore

    @process.register(CoefficientDerivative)
    @DAGTraverser.postorder_only_children([0])
    def _(self, o: CoefficientDerivative, f: Expr | BaseForm) -> Expr | BaseForm:
        """Apply to a coefficient_derivative."""
        _, w, v, cd = o.ufl_operands
        key = (GateauxDerivativeRuleset, w, v, cd)
        dag_traverser = self._dag_traverser_cache.setdefault(
            key,
            GateauxDerivativeRuleset(w, v, cd),  # type: ignore
        )
        return dag_traverser(f)  # type: ignore

    @process.register(BaseFormDerivative)
    @DAGTraverser.postorder_only_children([0])
    def _(self, o: BaseFormDerivative, f: BaseForm) -> Expr | BaseForm:
        """Apply to a base_form_derivative."""
        _, w, v, cd = o.ufl_operands
        key = (BaseFormDerivativeRuleset, w, v, cd)
        dag_traverser = self._dag_traverser_cache.setdefault(
            key,
            BaseFormDerivativeRuleset(w, v, cd),  # type: ignore
        )
        return dag_traverser(f)  # type: ignore

    @process.register(CoordinateDerivative)
    @DAGTraverser.postorder_only_children([0])
    def _(self, o: Expr, f: Expr | BaseForm) -> CoordinateDerivative:
        """Apply to a coordinate_derivative."""
        _, o1, o2, o3 = o.ufl_operands
        return CoordinateDerivative(f, o1, o2, o3)

    @process.register(BaseFormCoordinateDerivative)
    @DAGTraverser.postorder_only_children([0])
    def _(self, o: Expr, f: Expr | BaseForm) -> BaseFormCoordinateDerivative:
        """Apply to a base_form_coordinate_derivative."""
        _, o1, o2, o3 = o.ufl_operands
        return BaseFormCoordinateDerivative(f, o1, o2, o3)

    @process.register(Indexed)
    @DAGTraverser.postorder
    def _(self, o: Indexed, Ap: Expr, ii: MultiIndex) -> Expr | BaseForm:
        """Apply to an indexed."""
        # Reuse if untouched
        if Ap is o.ufl_operands[0]:
            return o
        r = len(Ap.ufl_shape) - len(ii)
        if r:
            kk = indices(r)
            op = Indexed(Ap, MultiIndex(ii.indices() + kk))
            op = as_tensor(op, kk)
        else:
            op = Indexed(Ap, ii)
        return op


def _uncontracted_arguments(N: BaseFormOperator) -> tuple[Argument, ...]:
    """Return the arguments of a base form operator that a form using it does not contract.

    A form contracts the dual argument of a base form operator in its integrand,
    N(u; v*) * v * dx = Action(v1 * v * dx, N(u; v*)), but not the arguments in its
    other slots, e.g. v in the Gateaux derivative dN/du[v] = dNdu(u; v, v*).
    """
    return tuple(
        a
        for slot in N.argument_slots(outer_form=True)
        for a in extract_type(slot, Argument, base_form_op_as_expr=True)
    )


def _last_operator(expressions) -> BaseFormOperator | None:
    """Return the last created base form operator with uncontracted arguments in expressions.

    An operator is created after the operators in its argument slots, so no
    other operator has the last one in its slots.
    """
    operators = [
        o
        for e in expressions
        for o in unique_pre_traversal(e)
        if isinstance(o, BaseFormOperator) and _uncontracted_arguments(o)
    ]
    return max(operators, key=lambda o: o.count(), default=None)


class BaseFormOperatorActionRestructurer(DAGTraverser):
    """Restructure a BaseForm into Actions on its base form operators with uncontracted arguments.

    A form F is linear in a base form operator N with an uncontracted argument
    v, such as the Gateaux derivative dN/du[v], so F is restructured as
    F = F|_{N=0} + Action(dF/dN, N). Since N is linear in its argument slots, N
    is restructured in the same way around a base form operator Q with
    uncontracted arguments in a slot. The result is an equal BaseForm whose integrands and argument
    slots contain no base form operators with uncontracted arguments.
    """

    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(Expr)
    @process.register(BaseForm)
    def _(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Apply to expr and base_form."""
        return o

    @process.register(Action)
    def _(self, o: Action) -> BaseForm:
        """Apply to action."""
        return self.reuse_if_untouched(o)

    @process.register(FormSum)
    def _(self, o: FormSum) -> BaseForm:
        """Apply to form_sum."""
        return FormSum(*((self(c), w) for c, w in zip(o.components(), o.weights())))

    @process.register(Form)
    def _(self, o: Form) -> BaseForm:
        """Apply to form."""
        N = _last_operator(itg.integrand() for itg in o.integrals())
        if N is None:
            return o
        term = self._contract(N, o)
        rest = replace(o, {N: Zero(N.ufl_shape)})
        if rest.empty():
            return term
        return self(rest) + term

    @process.register(BaseFormOperator)
    def _(self, o: BaseFormOperator) -> BaseForm:
        """Apply to base_form_operator."""
        vstar, *slots = o.argument_slots()
        N = _last_operator(slots)
        if N is None:
            return o
        term = self._contract(N, o)
        rest = [replace(slot, {N: Zero(N.ufl_shape)}) for slot in slots]
        if any(isinstance(slot, Zero) for slot in rest):
            return term
        rest = o._ufl_expr_reconstruct_(*o.ufl_operands, argument_slots=(vstar, *rest))
        return self(rest) + term

    def _contract(self, N: BaseFormOperator, F: BaseForm) -> BaseForm:
        """Return the Action contracting the dual argument of N with the argument Nhat of dF/dN.

        An Action contracts the last argument of its left operand with the first
        argument of its right operand, so Action(dF/dN, N) numbers the dual
        argument of N before its uncontracted arguments, and Action(N, dF/dN)
        numbers Nhat before the other arguments of dF/dN. The first is possible
        unless an uncontracted argument of N is numbered 0.
        """
        vstar, *slots = N.argument_slots()
        arguments = _uncontracted_arguments(N)
        numbers = [a.number() for a in arguments]
        adjoint = min(numbers) == 0
        if adjoint:
            Nhat_number, vstar_number = 0, 1 + max(numbers)
        else:
            others = [a.number() for a in F.arguments() if a not in arguments]
            Nhat_number, vstar_number = 1 + max(others, default=-1), 0
        Nhat = type(arguments[0])(vstar.ufl_function_space().dual(), Nhat_number)
        rules = GateauxDerivativeRuleset(ExprList(N), ExprList(Nhat), ExprMapping())
        dF_dN = self(map_integrands(rules, F))
        vstar = vstar.reconstruct(number=vstar_number)
        N = self(N._ufl_expr_reconstruct_(*N.ufl_operands, argument_slots=(vstar, *slots)))
        if adjoint:
            return Action(N, dF_dN)
        return Action(dF_dN, N)


def apply_derivatives(expression):
    """Apply derivatives to an expression.

    Args:
        expression: A Form, an Expr or a BaseFormOperator to be differentiated

    Returns:
        A differentiated expression
    """
    dexpression = map_integrands(DerivativeRuleDispatcher(), expression)
    if not isinstance(expression, BaseForm):
        return dexpression
    if isinstance(dexpression, int) or (
        isinstance(dexpression, Form) and dexpression.empty() and not expression.empty()
    ):
        # The derivative vanishes: keep its arguments, which an empty Form has lost.
        return ZeroBaseForm(expression.arguments())
    return BaseFormOperatorActionRestructurer()(dexpression)


class CoordinateDerivativeRuleset(GenericDerivativeRuleset):
    """Apply AFD (Automatic Functional Differentiation) to expression.

    Implements rules for the Gateaux derivative D_w[v](...) defined as
    D_w[v](e) = d/dtau e(w+tau v)|tau=0
    where 'e' is a ufl form after pullback and w is a SpatialCoordinate.
    """

    def __init__(
        self,
        coefficients: ExprList,
        arguments: ExprList,
        coefficient_derivatives: ExprMapping,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(
            (), compress=compress, visited_cache=visited_cache, result_cache=result_cache
        )
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

    # Work around singledispatchmethod inheritance issue;
    # see https://bugs.python.org/issue36457.
    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(GeometricQuantity)
    def _(self, o: Expr) -> Expr:
        # Explicitly defining dg/dw == 0
        return self.independent_terminal(o)

    @process.register(Argument)
    def _(self, o: Expr) -> Expr:
        # Explicitly defining da/dw == 0
        return self.independent_terminal(o)

    @process.register(Coefficient)
    def _(self, o: Expr) -> Expr:
        """Differentiate a coefficient."""
        raise NotImplementedError(
            "CoordinateDerivative of coefficient in physical space is not implemented."
        )

    @process.register(Grad)
    def _(self, o: Expr) -> Expr:
        """Differentiate a grad."""
        raise NotImplementedError("CoordinateDerivative grad in physical space is not implemented.")

    @process.register(SpatialCoordinate)
    def _(self, o: Expr) -> Expr:
        """Differentiate a spatial_coordinate."""
        do = self._w2v.get(o)  # type: ignore
        # d x /d x => Argument(x.function_space())
        if do is not None:
            return do
        else:
            raise NotImplementedError(
                "CoordinateDerivative found a SpatialCoordinate that is different "
                "from the one being differentiated."
            )

    @process.register(ReferenceValue)
    def _(self, o: Expr) -> Expr:
        """Differentiate a reference_value."""
        do = self._cd.get(o)  # type: ignore
        if do is not None:
            return do
        else:
            return self.independent_terminal(o)

    @process.register(ReferenceGrad)
    def _(self, g: Expr) -> Expr:
        """Differentiate a reference_grad."""
        # d (grad_X(...(x)) / dx => grad_X(...(Argument(x.function_space()))
        o = g
        ngrads = 0
        while isinstance(o, ReferenceGrad):
            (o,) = o.ufl_operands
            ngrads += 1
        if not (isinstance(o, SpatialCoordinate) or isinstance(o.ufl_operands[0], FormArgument)):
            raise ValueError(f"Expecting gradient of a FormArgument, not {ufl_err_str(o)}")

        def apply_grads(f):
            for i in range(ngrads):
                f = ReferenceGrad(f)
            return f

        # Find o among all w without any indexing, which makes this
        # easy
        for w, v in zip(self._w, self._v):
            if (
                o == w
                and isinstance(v, ReferenceValue)
                and isinstance(v.ufl_operands[0], FormArgument)
            ):
                # Case: d/dt [w + t v]
                return apply_grads(v)
        return self.independent_terminal(o)

    @process.register(Jacobian)
    def _(self, o: Expr) -> Expr:
        """Differentiate a jacobian."""
        # d (grad_X(x))/d x => grad_X(Argument(x.function_space())
        for w, v in zip(self._w, self._v):
            if extract_unique_domain(o) == extract_unique_domain(w) and isinstance(
                remove_complex_nodes(v).ufl_operands[0],  # type: ignore
                FormArgument,
            ):
                if isinstance(v, Conj):
                    return Conj(ReferenceGrad(remove_complex_nodes(v)))
                else:
                    return ReferenceGrad(v)
        return self.independent_terminal(o)


class CoordinateDerivativeRuleDispatcher(DAGTraverser):
    """Dispatcher."""

    def __init__(
        self,
        compress: bool | None = True,
        visited_cache: dict[tuple, Expr | BaseForm] | None = None,
        result_cache: dict[Expr | BaseForm, Expr | BaseForm] | None = None,
    ) -> None:
        """Initialise."""
        super().__init__(compress=compress, visited_cache=visited_cache, result_cache=result_cache)
        self._dag_traverser_cache: dict[
            tuple[type, Expr | BaseForm, Expr | BaseForm, Expr | BaseForm], DAGTraverser
        ] = {}

    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed object.

        """
        return super().process(o)

    @process.register(Expr)
    @process.register(BaseForm)  # type: ignore
    def _(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Apply to expr and base form."""
        return self.reuse_if_untouched(o)

    @process.register(Terminal)
    def _(self, o: Expr) -> Expr:
        """Apply to a terminal."""
        return o

    @process.register(Derivative)
    def _(self, o: Expr) -> Expr:
        """Apply to a derivative."""
        raise NotImplementedError(f"Missing derivative handler for {type(o).__name__}.")

    @process.register(Grad)
    def _(self, o: Expr) -> Expr:
        """Apply to a grad."""
        return o

    @process.register(ReferenceGrad)
    def _(self, o: Expr) -> Expr:
        """Apply to a reference_grad."""
        return o

    @process.register(CoefficientDerivative)
    def _(self, o: Expr) -> Expr:
        """Apply to a coefficient_derivative."""
        return o

    @process.register(CoordinateDerivative)
    @DAGTraverser.postorder_only_children([0])
    def _(self, o: Expr, f: Expr) -> Expr:
        """Apply to a coordinate_derivative."""
        from ufl.algorithms import extract_unique_elements

        for space in extract_unique_elements(o):
            if isinstance(space.pullback, CustomPullback):
                raise NotImplementedError(
                    "CoordinateDerivative is not supported for elements with custom pull back."
                )
        _, w, v, cd = o.ufl_operands
        key = (CoordinateDerivativeRuleset, w, v, cd)
        dag_traverser = self._dag_traverser_cache.setdefault(
            key,
            CoordinateDerivativeRuleset(w, v, cd),  # type: ignore
        )
        return dag_traverser(f)


def apply_coordinate_derivatives(expression):
    """Apply coordinate derivatives to an expression."""
    dag_traverser = CoordinateDerivativeRuleDispatcher()
    return map_integrands(dag_traverser, expression)
