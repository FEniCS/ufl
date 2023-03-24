# -*- coding: utf-8 -*-
"Differential operators."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009.

from ufl_legacy.log import error
from ufl_legacy.core.expr import Expr
from ufl_legacy.core.terminal import Terminal
from ufl_legacy.core.operator import Operator
from ufl_legacy.core.ufl_type import ufl_type

from ufl_legacy.exprcontainers import ExprList, ExprMapping
from ufl_legacy.constantvalue import Zero
from ufl_legacy.coefficient import Coefficient
from ufl_legacy.variable import Variable
from ufl_legacy.precedence import parstr
from ufl_legacy.domain import find_geometric_dimension
from ufl_legacy.checks import is_cellwise_constant


# --- Basic differentiation objects ---

@ufl_type(is_abstract=True,
          is_differential=True)
class Derivative(Operator):
    "Base class for all derivative types."
    __slots__ = ()

    def __init__(self, operands):
        Operator.__init__(self, operands)


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class CoefficientDerivative(Derivative):
    """Derivative of the integrand of a form w.r.t. the
    degrees of freedom in a discrete Coefficient."""
    __slots__ = ()

    def __new__(cls, integrand, coefficients, arguments,
                coefficient_derivatives):
        if not isinstance(coefficients, ExprList):
            error("Expecting ExprList instance with Coefficients.")
        if not isinstance(arguments, ExprList):
            error("Expecting ExprList instance with Arguments.")
        if not isinstance(coefficient_derivatives, ExprMapping):
            error("Expecting ExprMapping for coefficient derivatives.")
        if isinstance(integrand, Zero):
            return integrand
        return Derivative.__new__(cls)

    def __init__(self, integrand, coefficients, arguments,
                 coefficient_derivatives):
        if not isinstance(coefficient_derivatives, ExprMapping):
            coefficient_derivatives = ExprMapping(coefficient_derivatives)
        Derivative.__init__(self, (integrand, coefficients, arguments,
                                   coefficient_derivatives))

    def __str__(self):
        return "d/dfj { %s }, with fh=%s, dfh/dfj = %s, and coefficient derivatives %s"\
            % (self.ufl_operands[0], self.ufl_operands[1],
               self.ufl_operands[2], self.ufl_operands[3])


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class CoordinateDerivative(CoefficientDerivative):
    """Derivative of the integrand of a form w.r.t. the SpatialCoordinates."""
    __slots__ = ()

    def __str__(self):
        return "d/dfj { %s }, with fh=%s, dfh/dfj = %s, and coordinate derivatives %s"\
            % (self.ufl_operands[0], self.ufl_operands[1],
               self.ufl_operands[2], self.ufl_operands[3])


@ufl_type(num_ops=2)
class VariableDerivative(Derivative):
    __slots__ = (
        "ufl_shape",
        "ufl_free_indices",
        "ufl_index_dimensions",
    )

    def __new__(cls, f, v):
        # Checks
        if not isinstance(f, Expr):
            error("Expecting an Expr in VariableDerivative.")
        if not isinstance(v, (Variable, Coefficient)):
            error("Expecting a Variable in VariableDerivative.")
        if v.ufl_free_indices:
            error("Differentiation variable cannot have free indices.")

        # Simplification
        # Return zero if expression is trivially independent of variable
        if f._ufl_is_terminal_ and f != v:
            return Zero(f.ufl_shape + v.ufl_shape, f.ufl_free_indices,
                        f.ufl_index_dimensions)

        # Construction
        return Derivative.__new__(cls)

    def __init__(self, f, v):
        Derivative.__init__(self, (f, v))
        self.ufl_free_indices = f.ufl_free_indices
        self.ufl_index_dimensions = f.ufl_index_dimensions
        self.ufl_shape = f.ufl_shape + v.ufl_shape

    def __str__(self):
        if isinstance(self.ufl_operands[0], Terminal):
            return "d%s/d[%s]" % (self.ufl_operands[0], self.ufl_operands[1])
        return "d/d[%s] %s" % (self.ufl_operands[1],
                               parstr(self.ufl_operands[0], self))


# --- Compound differentiation objects ---

@ufl_type(is_abstract=True)
class CompoundDerivative(Derivative):
    "Base class for all compound derivative types."
    __slots__ = ()

    def __init__(self, operands):
        Derivative.__init__(self, operands)


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True)
class Grad(CompoundDerivative):
    __slots__ = ("_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            dim = find_geometric_dimension(f)
            return Zero(f.ufl_shape + (dim,), f.ufl_free_indices,
                        f.ufl_index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        self._dim = find_geometric_dimension(f)

    def _ufl_expr_reconstruct_(self, op):
        "Return a new object of the same type with new operands."
        if is_cellwise_constant(op):
            if op.ufl_shape != self.ufl_operands[0].ufl_shape:
                error("Operand shape mismatch in Grad reconstruct.")
            if self.ufl_operands[0].ufl_free_indices != op.ufl_free_indices:
                error("Free index mismatch in Grad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices,
                        self.ufl_index_dimensions)
        return self._ufl_class_(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        component, i = component[:-1], component[-1]
        derivatives = derivatives + (i,)
        result = self.ufl_operands[0].evaluate(x, mapping, component,
                                               index_values,
                                               derivatives=derivatives)
        return result

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape + (self._dim,)

    def __str__(self):
        return "grad(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True,
          is_in_reference_frame=True)
class ReferenceGrad(CompoundDerivative):
    __slots__ = ("_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            dim = f.ufl_domain().topological_dimension()
            return Zero(f.ufl_shape + (dim,), f.ufl_free_indices,
                        f.ufl_index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        self._dim = f.ufl_domain().topological_dimension()

    def _ufl_expr_reconstruct_(self, op):
        "Return a new object of the same type with new operands."
        if is_cellwise_constant(op):
            if op.ufl_shape != self.ufl_operands[0].ufl_shape:
                error("Operand shape mismatch in ReferenceGrad reconstruct.")
            if self.ufl_operands[0].ufl_free_indices != op.ufl_free_indices:
                error("Free index mismatch in ReferenceGrad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices,
                        self.ufl_index_dimensions)
        return self._ufl_class_(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        "Get child from mapping and return the component asked for."
        component, i = component[:-1], component[-1]
        derivatives = derivatives + (i,)
        result = self.ufl_operands[0].evaluate(x, mapping, component,
                                               index_values,
                                               derivatives=derivatives)
        return result

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape + (self._dim,)

    def __str__(self):
        return "reference_grad(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True)
class Div(CompoundDerivative):
    __slots__ = ()

    def __new__(cls, f):
        if f.ufl_free_indices:
            error("Free indices in the divergence argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            return Zero(f.ufl_shape[:-1])  # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape[:-1]

    def __str__(self):
        return "div(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True,
          is_in_reference_frame=True)
class ReferenceDiv(CompoundDerivative):
    __slots__ = ()

    def __new__(cls, f):
        if f.ufl_free_indices:
            error("Free indices in the divergence argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            return Zero(f.ufl_shape[:-1])  # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape[:-1]

    def __str__(self):
        return "reference_div(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class NablaGrad(CompoundDerivative):
    __slots__ = ("_dim",)

    def __new__(cls, f):
        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            dim = find_geometric_dimension(f)
            return Zero((dim,) + f.ufl_shape, f.ufl_free_indices,
                        f.ufl_index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))
        self._dim = find_geometric_dimension(f)

    def _ufl_expr_reconstruct_(self, op):
        "Return a new object of the same type with new operands."
        if is_cellwise_constant(op):
            if op.ufl_shape != self.ufl_operands[0].ufl_shape:
                error("Operand shape mismatch in NablaGrad reconstruct.")
            if self.ufl_operands[0].ufl_free_indices != op.ufl_free_indices:
                error("Free index mismatch in NablaGrad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices,
                        self.ufl_index_dimensions)
        return self._ufl_class_(op)

    @property
    def ufl_shape(self):
        return (self._dim,) + self.ufl_operands[0].ufl_shape

    def __str__(self):
        return "nabla_grad(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class NablaDiv(CompoundDerivative):
    __slots__ = ()

    def __new__(cls, f):
        if f.ufl_free_indices:
            error("Free indices in the divergence argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            return Zero(f.ufl_shape[1:])  # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        return self.ufl_operands[0].ufl_shape[1:]

    def __str__(self):
        return "nabla_div(%s)" % self.ufl_operands[0]


_curl_shapes = {(): (2,), (2,): (), (3,): (3,)}


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True)
class Curl(CompoundDerivative):
    __slots__ = ("ufl_shape",)

    def __new__(cls, f):
        # Validate input
        sh = f.ufl_shape
        if f.ufl_shape not in ((), (2,), (3,)):
            error("Expecting a scalar, 2D vector or 3D vector.")
        if f.ufl_free_indices:
            error("Free indices in the curl argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            sh = {(): (2,), (2,): (), (3,): (3,)}[sh]
            return Zero(sh)  # No free indices asserted above
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        global _curl_shapes
        CompoundDerivative.__init__(self, (f,))
        self.ufl_shape = _curl_shapes[f.ufl_shape]

    def __str__(self):
        return "curl(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0,
          is_terminal_modifier=True, is_in_reference_frame=True)
class ReferenceCurl(CompoundDerivative):
    __slots__ = ("ufl_shape",)

    def __new__(cls, f):
        # Validate input
        sh = f.ufl_shape
        if f.ufl_shape not in ((), (2,), (3,)):
            error("Expecting a scalar, 2D vector or 3D vector.")
        if f.ufl_free_indices:
            error("Free indices in the curl argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            sh = {(): (2,), (2,): (), (3,): (3,)}[sh]
            return Zero(sh)  # No free indices asserted above
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        global _curl_shapes
        CompoundDerivative.__init__(self, (f,))
        self.ufl_shape = _curl_shapes[f.ufl_shape]

    def __str__(self):
        return "reference_curl(%s)" % self.ufl_operands[0]
