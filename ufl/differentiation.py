"""Differential operators."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.checks import is_cellwise_constant
from ufl.coefficient import Coefficient
from ufl.argument import Argument, Coargument
from ufl.constantvalue import Zero
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.terminal import Terminal
from ufl.core.ufl_type import ufl_type
from ufl.domain import extract_unique_domain, find_geometric_dimension
from ufl.exprcontainers import ExprList, ExprMapping
from ufl.form import BaseForm
from ufl.precedence import parstr
from ufl.variable import Variable

# --- Basic differentiation objects ---


@ufl_type(is_abstract=True,
          is_differential=True)
class Derivative(Operator):
    """Base class for all derivative types."""

    __slots__ = ()

    def __init__(self, operands):
        """Initalise."""
        Operator.__init__(self, operands)


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class CoefficientDerivative(Derivative):
    """Derivative of the integrand of a form w.r.t. the degrees of freedom in a discrete Coefficient."""

    __slots__ = ()

    def __new__(cls, integrand, coefficients, arguments,
                coefficient_derivatives):
        """Create a new CoefficientDerivative."""
        if not isinstance(coefficients, ExprList):
            raise ValueError("Expecting ExprList instance with Coefficients.")
        if not isinstance(arguments, ExprList):
            raise ValueError("Expecting ExprList instance with Arguments.")
        if not isinstance(coefficient_derivatives, ExprMapping):
            raise ValueError("Expecting ExprMapping for coefficient derivatives.")
        if isinstance(integrand, Zero):
            return integrand
        return Derivative.__new__(cls)

    def __init__(self, integrand, coefficients, arguments,
                 coefficient_derivatives):
        """Initalise."""
        if not isinstance(coefficient_derivatives, ExprMapping):
            coefficient_derivatives = ExprMapping(coefficient_derivatives)
        Derivative.__init__(self, (integrand, coefficients, arguments,
                                   coefficient_derivatives))

    def __str__(self):
        """Format as a string."""
        return "d/dfj { %s }, with fh=%s, dfh/dfj = %s, and coefficient derivatives %s"\
            % (self.ufl_operands[0], self.ufl_operands[1],
               self.ufl_operands[2], self.ufl_operands[3])


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class CoordinateDerivative(CoefficientDerivative):
    """Derivative of the integrand of a form w.r.t. the SpatialCoordinates."""

    __slots__ = ()

    def __str__(self):
        """Format as a string."""
        return "d/dfj { %s }, with fh=%s, dfh/dfj = %s, and coordinate derivatives %s"\
            % (self.ufl_operands[0], self.ufl_operands[1],
               self.ufl_operands[2], self.ufl_operands[3])


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class BaseFormDerivative(CoefficientDerivative, BaseForm):
    """Derivative of a base form w.r.t the degrees of freedom in a discrete Coefficient."""

    _ufl_noslots_ = True

    def __init__(self, base_form, coefficients, arguments,
                 coefficient_derivatives):
        """Initalise."""
        CoefficientDerivative.__init__(self, base_form, coefficients, arguments,
                                       coefficient_derivatives)
        BaseForm.__init__(self)

    def _analyze_form_arguments(self):
        """Collect the arguments of the corresponding BaseForm."""
        from ufl.algorithms.analysis import extract_type, extract_coefficients
        base_form, _, arguments, _ = self.ufl_operands

        def arg_type(x):
            if isinstance(x, BaseForm):
                return Coargument
            return Argument
        # Each derivative arguments can either be a:
        # - `ufl.BaseForm`: if it contains a `ufl.Coargument`
        # - or a `ufl.Expr`: if it contains a `ufl.Argument`
        # When a `Coargument` is encountered, it is treated as an argument (i.e. as V* -> V* and not V* x V -> R)
        # and should result in one single argument (in the dual space).
        base_form_args = base_form.arguments() + tuple(arg for a in arguments.ufl_operands
                                                       for arg in extract_type(a, arg_type(a)))
        # BaseFormDerivative's arguments don't necessarily contain BaseArgument objects only
        # -> e.g. `derivative(u ** 2, u, u)` with `u` a Coefficient.
        base_form_coeffs = base_form.coefficients() + tuple(arg for a in arguments.ufl_operands
                                                            for arg in extract_coefficients(a))
        # Reconstruct arguments for correct numbering
        self._arguments = tuple(type(arg)(arg.ufl_function_space(), arg.number(), arg.part()) for arg in base_form_args)
        self._coefficients = base_form_coeffs


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class BaseFormCoordinateDerivative(BaseFormDerivative, CoordinateDerivative):
    """Derivative of a base form w.r.t. the SpatialCoordinates."""

    _ufl_noslots_ = True

    def __init__(self, base_form, coefficients, arguments,
                 coefficient_derivatives):
        """Initalise."""
        BaseFormDerivative.__init__(self, base_form, coefficients, arguments,
                                    coefficient_derivatives)


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class BaseFormOperatorDerivative(BaseFormDerivative, BaseFormOperator):
    """Derivative of a base form operator w.r.t the degrees of freedom in a discrete Coefficient."""
    _ufl_noslots_ = True

    # BaseFormOperatorDerivative is only needed because of a different
    # differentiation procedure for BaseformOperator objects.
    def __init__(self, base_form, coefficients, arguments,
                 coefficient_derivatives):
        """Initalise."""
        BaseFormDerivative.__init__(self, base_form, coefficients, arguments,
                                    coefficient_derivatives)
        self._argument_slots = base_form._argument_slots

    # Enforce Operator reconstruction as Operator is a parent class of both: BaseFormDerivative and BaseFormOperator.
    # Therfore the latter overwrites Operator reconstruction and we would have:
    #   -> BaseFormOperatorDerivative._ufl_expr_reconstruct_ = BaseFormOperator._ufl_expr_reconstruct_
    _ufl_expr_reconstruct_ = Operator._ufl_expr_reconstruct_
    # Set __repr__
    __repr__ = Operator.__repr__

    def argument_slots(self, outer_form=False):
        """Return a tuple of expressions containing argument and coefficient based expressions."""
        from ufl.algorithms.analysis import extract_arguments
        base_form, _, arguments, _ = self.ufl_operands
        argument_slots = (base_form.argument_slots(outer_form)
                          + tuple(arg for a in arguments for arg in extract_arguments(a)))
        return argument_slots


@ufl_type(num_ops=4, inherit_shape_from_operand=0,
          inherit_indices_from_operand=0)
class BaseFormOperatorCoordinateDerivative(BaseFormOperatorDerivative, CoordinateDerivative):
    """Derivative of a base form operator w.r.t. the SpatialCoordinates."""
    _ufl_noslots_ = True

    def __init__(self, base_form, coefficients, arguments,
                 coefficient_derivatives):
        """Initalise."""
        BaseFormOperatorDerivative.__init__(self, base_form, coefficients, arguments,
                                            coefficient_derivatives)


@ufl_type(num_ops=2)
class VariableDerivative(Derivative):
    """Variable Derivative."""

    __slots__ = (
        "ufl_shape",
        "ufl_free_indices",
        "ufl_index_dimensions",
    )

    def __new__(cls, f, v):
        """Create a new VariableDerivative."""
        # Checks
        if not isinstance(f, Expr):
            raise ValueError("Expecting an Expr in VariableDerivative.")
        if not isinstance(v, (Variable, Coefficient)):
            raise ValueError("Expecting a Variable in VariableDerivative.")
        if v.ufl_free_indices:
            raise ValueError("Differentiation variable cannot have free indices.")

        # Simplification
        # Return zero if expression is trivially independent of variable
        if f._ufl_is_terminal_ and f != v:
            return Zero(f.ufl_shape + v.ufl_shape, f.ufl_free_indices,
                        f.ufl_index_dimensions)

        # Construction
        return Derivative.__new__(cls)

    def __init__(self, f, v):
        """Initalise."""
        Derivative.__init__(self, (f, v))
        self.ufl_free_indices = f.ufl_free_indices
        self.ufl_index_dimensions = f.ufl_index_dimensions
        self.ufl_shape = f.ufl_shape + v.ufl_shape

    def __str__(self):
        """Format as a string."""
        if isinstance(self.ufl_operands[0], Terminal):
            return "d%s/d[%s]" % (self.ufl_operands[0], self.ufl_operands[1])
        return "d/d[%s] %s" % (self.ufl_operands[1],
                               parstr(self.ufl_operands[0], self))


# --- Compound differentiation objects ---

@ufl_type(is_abstract=True)
class CompoundDerivative(Derivative):
    """Base class for all compound derivative types."""

    __slots__ = ()

    def __init__(self, operands):
        """Initalise."""
        Derivative.__init__(self, operands)


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True)
class Grad(CompoundDerivative):
    """Grad."""

    __slots__ = ("_dim",)

    def __new__(cls, f):
        """Create a new Grad."""
        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            dim = find_geometric_dimension(f)
            return Zero(f.ufl_shape + (dim,), f.ufl_free_indices,
                        f.ufl_index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))
        self._dim = find_geometric_dimension(f)

    def _ufl_expr_reconstruct_(self, op):
        """Return a new object of the same type with new operands."""
        if is_cellwise_constant(op):
            if op.ufl_shape != self.ufl_operands[0].ufl_shape:
                raise ValueError("Operand shape mismatch in Grad reconstruct.")
            if self.ufl_operands[0].ufl_free_indices != op.ufl_free_indices:
                raise ValueError("Free index mismatch in Grad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices,
                        self.ufl_index_dimensions)
        return self._ufl_class_(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        """Get child from mapping and return the component asked for."""
        component, i = component[:-1], component[-1]
        derivatives = derivatives + (i,)
        result = self.ufl_operands[0].evaluate(x, mapping, component,
                                               index_values,
                                               derivatives=derivatives)
        return result

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape + (self._dim,)

    def __str__(self):
        """Format as a string."""
        return "grad(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True,
          is_in_reference_frame=True)
class ReferenceGrad(CompoundDerivative):
    """Reference grad."""

    __slots__ = ("_dim", )

    def __new__(cls, f):
        """Create a new ReferenceGrad."""
        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            dim = extract_unique_domain(f).topological_dimension()
            return Zero(f.ufl_shape + (dim,), f.ufl_free_indices,
                        f.ufl_index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))
        self._dim = extract_unique_domain(f).topological_dimension()

    def _ufl_expr_reconstruct_(self, op):
        """Return a new object of the same type with new operands."""
        if is_cellwise_constant(op):
            if op.ufl_shape != self.ufl_operands[0].ufl_shape:
                raise ValueError("Operand shape mismatch in ReferenceGrad reconstruct.")
            if self.ufl_operands[0].ufl_free_indices != op.ufl_free_indices:
                raise ValueError("Free index mismatch in ReferenceGrad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices,
                        self.ufl_index_dimensions)
        return self._ufl_class_(op)

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        """Get child from mapping and return the component asked for."""
        component, i = component[:-1], component[-1]
        derivatives = derivatives + (i,)
        result = self.ufl_operands[0].evaluate(x, mapping, component,
                                               index_values,
                                               derivatives=derivatives)
        return result

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape + (self._dim,)

    def __str__(self):
        """Format as a string."""
        return "reference_grad(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True)
class Div(CompoundDerivative):
    """Div."""

    __slots__ = ()

    def __new__(cls, f):
        """Create a new Div."""
        if f.ufl_free_indices:
            raise ValueError("Free indices in the divergence argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            return Zero(f.ufl_shape[:-1])  # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape[:-1]

    def __str__(self):
        """Format as a string."""
        return "div(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True,
          is_in_reference_frame=True)
class ReferenceDiv(CompoundDerivative):
    """Reference divergence."""

    __slots__ = ()

    def __new__(cls, f):
        """Create a new ReferenceDiv."""
        if f.ufl_free_indices:
            raise ValueError("Free indices in the divergence argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            return Zero(f.ufl_shape[:-1])  # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape[:-1]

    def __str__(self):
        """Format as a string."""
        return "reference_div(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class NablaGrad(CompoundDerivative):
    """Nabla grad."""

    __slots__ = ("_dim",)

    def __new__(cls, f):
        """Create a new NablaGrad."""
        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            dim = find_geometric_dimension(f)
            return Zero((dim,) + f.ufl_shape, f.ufl_free_indices,
                        f.ufl_index_dimensions)
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))
        self._dim = find_geometric_dimension(f)

    def _ufl_expr_reconstruct_(self, op):
        """Return a new object of the same type with new operands."""
        if is_cellwise_constant(op):
            if op.ufl_shape != self.ufl_operands[0].ufl_shape:
                raise ValueError("Operand shape mismatch in NablaGrad reconstruct.")
            if self.ufl_operands[0].ufl_free_indices != op.ufl_free_indices:
                raise ValueError("Free index mismatch in NablaGrad reconstruct.")
            return Zero(self.ufl_shape, self.ufl_free_indices,
                        self.ufl_index_dimensions)
        return self._ufl_class_(op)

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return (self._dim,) + self.ufl_operands[0].ufl_shape

    def __str__(self):
        """Format as a string."""
        return "nabla_grad(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0)
class NablaDiv(CompoundDerivative):
    """Nabla div."""

    __slots__ = ()

    def __new__(cls, f):
        """Create a new NablaDiv."""
        if f.ufl_free_indices:
            raise ValueError("Free indices in the divergence argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            return Zero(f.ufl_shape[1:])  # No free indices asserted above

        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return self.ufl_operands[0].ufl_shape[1:]

    def __str__(self):
        """Format as a string."""
        return "nabla_div(%s)" % self.ufl_operands[0]


_curl_shapes = {(): (2,), (2,): (), (3,): (3,)}


@ufl_type(num_ops=1, inherit_indices_from_operand=0, is_terminal_modifier=True)
class Curl(CompoundDerivative):
    """Compound derivative."""

    __slots__ = ("ufl_shape",)

    def __new__(cls, f):
        """Create a new CompoundDerivative."""
        # Validate input
        sh = f.ufl_shape
        if f.ufl_shape not in ((), (2,), (3,)):
            raise ValueError("Expecting a scalar, 2D vector or 3D vector.")
        if f.ufl_free_indices:
            raise ValueError("Free indices in the curl argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            sh = {(): (2,), (2,): (), (3,): (3,)}[sh]
            return Zero(sh)  # No free indices asserted above
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))
        self.ufl_shape = _curl_shapes[f.ufl_shape]

    def __str__(self):
        """Format as a string."""
        return "curl(%s)" % self.ufl_operands[0]


@ufl_type(num_ops=1, inherit_indices_from_operand=0,
          is_terminal_modifier=True, is_in_reference_frame=True)
class ReferenceCurl(CompoundDerivative):
    """Reference curl."""

    __slots__ = ("ufl_shape",)

    def __new__(cls, f):
        """Create a new ReferenceCurl."""
        # Validate input
        sh = f.ufl_shape
        if f.ufl_shape not in ((), (2,), (3,)):
            raise ValueError("Expecting a scalar, 2D vector or 3D vector.")
        if f.ufl_free_indices:
            raise ValueError("Free indices in the curl argument is not allowed.")

        # Return zero if expression is trivially constant
        if is_cellwise_constant(f):
            sh = {(): (2,), (2,): (), (3,): (3,)}[sh]
            return Zero(sh)  # No free indices asserted above
        return CompoundDerivative.__new__(cls)

    def __init__(self, f):
        """Initalise."""
        CompoundDerivative.__init__(self, (f,))
        self.ufl_shape = _curl_shapes[f.ufl_shape]

    def __str__(self):
        """Format as a string."""
        return "reference_curl(%s)" % self.ufl_operands[0]
