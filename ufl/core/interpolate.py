"""This module defines the Interpolate class."""

# Copyright (C) 2021 Nacime Bouziani
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Nacime Bouziani, 2021-2022

import hashlib
from collections import defaultdict
from itertools import chain

from ufl.argument import Argument, Coargument
from ufl.coefficient import Cofunction
from ufl.constantvalue import as_ufl
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.ufl_type import ufl_type
from ufl.duals import is_dual
from ufl.finiteelement import AbstractFiniteElement
from ufl.form import BaseForm
from ufl.functionspace import AbstractFunctionSpace


@ufl_type(num_ops="varying", is_differential=True)
class Interpolate(BaseFormOperator):
    """Symbolic representation of the interpolation operator."""

    # Slots are disabled here because they cause trouble in PyDOLFIN
    # multiple inheritance pattern:
    _ufl_noslots_ = True

    def __init__(self, expr, v):
        """Initialise.

        Args:
            expr: a UFL expression to interpolate.
            v: the FunctionSpace to interpolate into or the Coargument
                defined on the dual of the FunctionSpace to interpolate into.
        """
        from ufl.algorithms import extract_arguments

        expr = as_ufl(expr)

        if isinstance(expr, BaseForm):
            expr_args = expr.arguments()
            if not isinstance(expr_args[0], Coargument):
                raise ValueError("Expecting the first argument to be primal.")
            expr_args = expr_args[1:]
        else:
            expr_args = extract_arguments(expr)
        expr_arg_numbers = {arg.number() for arg in expr_args}

        if isinstance(v, AbstractFunctionSpace):
            if is_dual(v):
                raise ValueError("Expecting a primal function space.")
            n = 1 if expr_arg_numbers == {0} else 0
            v = Argument(v.dual(), n)
            dual_arg_numbers = {n}
        elif isinstance(v, BaseForm):
            dual_arg_numbers = {arg.number() for arg in v.arguments() if is_dual(arg)}
        else:
            raise ValueError("Expecting the second argument to be FunctionSpace or BaseForm.")

        # Check valid argument numbering
        if expr_arg_numbers & dual_arg_numbers:
            raise ValueError("Same argument numbers in first and second operands to interpolate.")
        if expr_arg_numbers | dual_arg_numbers not in [set(), {0}, {0, 1}]:
            raise ValueError("Non-contiguous argument numbers in interpolate.")

        # Reversed order convention
        argument_slots = (v, expr)
        # Get the primal space (V** = V)
        function_space = v.arguments()[0].ufl_function_space()
        self._function_space = function_space

        # Set the operand as `expr` for DAG traversal purpose.
        operand = expr
        BaseFormOperator.__init__(
            self, operand, function_space=function_space, argument_slots=argument_slots
        )
        self._cache = {}
        self._domains = None
        self._signature = None
        self._subdomain_data = None
        self._terminal_numbering = None

    def _analyze_form_arguments(self) -> None:
        """Analyze arguments and coefficients in the interpolation."""
        from ufl.algorithms.analysis import extract_coefficients

        super()._analyze_form_arguments()
        self._coefficients = tuple(extract_coefficients(self))

    def _analyze_domains(self) -> None:
        """Analyze domains in the interpolation and its argument slots."""
        from ufl.domain import extract_domains, join_domains, sort_domains

        def extract(expression):
            if isinstance(expression, BaseForm):
                return expression.ufl_domains()
            return extract_domains(expression)

        expressions = (*self.ufl_operands, *self.argument_slots())
        self._domains = sort_domains(
            join_domains(chain.from_iterable(extract(e) for e in expressions))
        )

    def ufl_domains(self):
        """Return all domains found in the interpolation."""
        if self._domains is None:
            self._analyze_domains()
        return self._domains

    def subdomain_data(self):
        """Return cell-iteration subdomain data for the target domain."""
        if self._subdomain_data is None:
            domain = self._function_space.ufl_domain()
            self._subdomain_data = {domain: {"cell": [None]}}
        return self._subdomain_data

    def terminal_numbering(self):
        """Return a contiguous numbering for counted interpolation objects."""
        from ufl.algorithms.analysis import extract_type
        from ufl.utils.counted import Counted
        from ufl.utils.sorting import sorted_by_count

        if self._terminal_numbering is None:
            exprs_by_type = defaultdict(set)
            for counted_expr in extract_type(self, Counted):
                exprs_by_type[counted_expr._counted_class].add(counted_expr)

            numbering = {
                expression: i for i, expression in enumerate(self.arguments())
            }
            numbering.update(
                {
                    expression: i
                    for i, expression in enumerate(self.coefficients())
                }
            )
            for expressions in exprs_by_type.values():
                for i, expression in enumerate(sorted_by_count(expressions)):
                    numbering.setdefault(expression, i)
            self._terminal_numbering = numbering
        return self._terminal_numbering

    def signature(self):
        """Return a numbering-independent signature for compiler caches."""
        from ufl.algorithms.signature import compute_expression_signature
        from ufl.form import Form, FormSum

        if self._signature is None:
            renumbering = {domain: i for i, domain in enumerate(self.ufl_domains())}
            renumbering.update(self.terminal_numbering())

            def signature(slot):
                if isinstance(slot, Interpolate):
                    return "Interpolate", slot.signature()
                if isinstance(slot, Form):
                    return "Form", slot.signature()
                if isinstance(slot, FormSum):
                    return "FormSum", tuple(
                        (signature(component), signature(as_ufl(weight)))
                        for component, weight in zip(
                            slot.components(), slot.weights()
                        )
                    )
                if isinstance(slot, Coargument | Cofunction):
                    kind = type(slot).__name__
                    slot = Argument(slot.ufl_function_space().dual(), 0)
                    renumbering[slot] = 0
                    return kind, compute_expression_signature(slot, renumbering)
                if isinstance(slot, BaseForm):
                    return type(slot).__name__, tuple(
                        signature(operand) for operand in slot.ufl_operands
                    )
                return compute_expression_signature(slot, renumbering)

            signatures = tuple(
                signature(slot) for slot in self.argument_slots()
            )
            self._signature = hashlib.sha512(str(signatures).encode("utf-8")).hexdigest()
        return self._signature

    def ufl_element(self) -> AbstractFiniteElement:
        """Return the target finite element."""
        return self._function_space.ufl_element()

    def _ufl_expr_reconstruct_(self, expr, v=None, **add_kwargs):
        """Return a new object of the same type with new operands."""
        v = v or self.argument_slots()[0]
        return type(self)(expr, v, **add_kwargs)

    def __repr__(self):
        """Default repr string construction for Interpolate."""
        r = "Interpolate("
        r += ", ".join(repr(arg) for arg in reversed(self.argument_slots()))
        r += f"; {self.ufl_function_space()!r})"
        return r

    def __str__(self):
        """Default str string construction for Interpolate."""
        s = "Interpolate("
        s += ", ".join(str(arg) for arg in reversed(self.argument_slots()))
        s += f"; {self.ufl_function_space()})"
        return s

    def __eq__(self, other):
        """Check for equality."""
        if self is other:
            return True
        return (
            type(self) is type(other)
            and all(a == b for a, b in zip(self._argument_slots, other._argument_slots))
            and self.ufl_function_space() == other.ufl_function_space()
        )


# Helper function
def interpolate(expr, v):
    """Create symbolic representation of the interpolation operator.

    Args:
        expr:
            a UFL expression to interpolate.
        v:
            the FunctionSpace to interpolate into or the Coargument
            defined on the dual of the FunctionSpace to interpolate into.

    """
    return Interpolate(expr, v)
