"""Algorithm for replacing gradients in an expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from __future__ import annotations

from functools import singledispatchmethod

from ufl.algorithms.map_integrands import map_integrands
from ufl.classes import Argument, Coefficient, Expr, FormArgument, Interpolate, ReferenceValue
from ufl.corealg.dag_traverser import DAGTraverser
from ufl.domain import extract_unique_domain
from ufl.form import BaseForm


class FunctionPullbackApplier(DAGTraverser):
    """A pull back applier."""

    @singledispatchmethod
    def process(self, o: Expr) -> Expr:
        """Process ``o``.

        Args:
            o: `Expr` to be processed.

        Returns:
            Processed `Expr`.

        """
        return super().process(o)

    @process.register(Expr)
    def _(self, o: Expr) -> Expr:
        """Handle Expr."""
        return self.reuse_if_untouched(o)

    @process.register(FormArgument)
    def _(self, o: Argument | Coefficient) -> Expr:
        """Represent 0-derivatives of a form argument on the reference element."""
        r = ReferenceValue(o)
        space = o.ufl_function_space()
        element = o.ufl_element()

        if r.ufl_shape != element.reference_value_shape:
            raise ValueError(
                "Expecting reference space expression with shape "
                f"'{element.reference_value_shape}', got '{r.ufl_shape}'"
            )
        f = element.pullback.apply(r)
        if f.ufl_shape != space.value_shape:
            raise ValueError(
                f"Expecting pulled back expression with shape '{space.value_shape}', "
                f"got '{f.ufl_shape}'"
            )

        assert f.ufl_shape == o.ufl_shape
        return f


class InterpolatePullbackApplier(DAGTraverser):
    """A pull back applier for interpolation."""

    @singledispatchmethod
    def process(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Process ``o``.

        Args:
            o: `Expr` or `BaseForm` to be processed.

        Returns:
            Processed `Expr` or `BaseForm`.

        """
        return super().process(o)

    @process.register(Expr)
    @process.register(BaseForm)
    def _(self, o: Expr | BaseForm) -> Expr | BaseForm:
        """Handle Expr and BaseForm."""
        return self.reuse_if_untouched(o)

    @process.register(Interpolate)
    @DAGTraverser.postorder
    def _(self, o: Interpolate, operand: Expr) -> Expr:
        """Evaluate an interpolation on the reference cell of its target element."""
        dual_arg, _ = o.argument_slots()
        element = o.ufl_element()
        domain = extract_unique_domain(operand) or dual_arg.ufl_function_space().ufl_domain()
        # Build the node here rather than reconstructing o: the mapped operand
        # no longer has the physical value shape that a subclass may check.
        r = Interpolate(apply_inverse_pullback(operand, element, domain), dual_arg)
        return element.pullback.apply(ReferenceValue(r), domain)


def apply_inverse_pullback(expr, element, domain=None):
    """Map a physical expression onto the reference cell of an element.

    This is a rule on a single node, not a DAG traversal: the expression is
    mapped as a whole, and an interpolation inside it is left alone for
    `apply_interpolate_pullbacks` to lower.

    Args:
        expr: An expression on a physical cell, whose shape must be the
            physical value shape of the element
        element: The element whose pull back is inverted
        domain: The domain to use if the expression carries none

    Returns:
        The expression on the reference cell, with shape
        ``element.reference_value_shape``
    """
    mesh = extract_unique_domain(expr) or domain
    if domain is not None and mesh != domain:
        raise NotImplementedError("Multiple domains not supported")
    physical_value_shape = element.pullback.physical_value_shape(element, mesh)
    if expr.ufl_shape != physical_value_shape:
        raise ValueError(
            f"Expecting physical expression with shape '{physical_value_shape}', "
            f"got '{expr.ufl_shape}'"
        )
    r = element.pullback.apply_inverse(expr, mesh)
    if r.ufl_shape != element.reference_value_shape:
        raise ValueError(
            f"Expecting reference expression with shape "
            f"'{element.reference_value_shape}', got '{r.ufl_shape}'"
        )
    return r


def apply_interpolate_pullbacks(expr):
    """Change the representation of the interpolations in an expression.

    An interpolation is evaluated on the reference cell of its target element,
    so its operand is mapped there and the result is pulled back to the
    physical cell for the expression that holds it.

    Args:
        expr: An Expr or Form

    Returns:
        The expression with its interpolations on their reference cells
    """
    return map_integrands(InterpolatePullbackApplier(), expr)


def apply_function_pullbacks(expr):
    """Change representation of coefficients and arguments in an expression.

    Applies Piola mappings where applicable and represents all
    form arguments in reference value.

    Args:
        expr: An Expression
    """
    return map_integrands(FunctionPullbackApplier(), expr)
