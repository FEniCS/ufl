"""Algorithm for replacing terminals in an expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

from ufl.classes import CoefficientDerivative, Interpolate, ExternalOperator, Form
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.analysis import has_exact_type


class Replacer(MultiFunction):
    """Replacer."""

    def __init__(self, mapping):
        """Initialize."""
        super().__init__()
        self.mapping = mapping

        # One can replace Coarguments by 1-Forms
        def get_shape(x):
            """Get the shape of an object."""
            if isinstance(x, Form):
                return x.arguments()[0].ufl_shape
            return x.ufl_shape

        if not all(get_shape(k) == get_shape(v) for k, v in mapping.items()):
            raise ValueError("Replacement expressions must have the same shape as what they replace.")

    def ufl_type(self, o, *args):
        """Replace a ufl_type."""
        try:
            return self.mapping[o]
        except KeyError:
            return self.reuse_if_untouched(o, *args)

    def external_operator(self, o):
        """Replace an external_operator."""
        o = self.mapping.get(o) or o
        if isinstance(o, ExternalOperator):
            new_ops = tuple(replace(op, self.mapping) for op in o.ufl_operands)
            new_args = tuple(replace(arg, self.mapping) for arg in o.argument_slots())
            return o._ufl_expr_reconstruct_(*new_ops, argument_slots=new_args)
        return o

    def interpolate(self, o):
        """Replace an interpolate."""
        o = self.mapping.get(o) or o
        if isinstance(o, Interpolate):
            new_args = tuple(replace(arg, self.mapping) for arg in o.argument_slots())
            return o._ufl_expr_reconstruct_(*reversed(new_args))
        return o

    def coefficient_derivative(self, o):
        """Replace a coefficient derivative."""
        raise ValueError("Derivatives should be applied before executing replace.")


def replace(e, mapping):
    """Replace subexpressions in expression.

    Params:
        e: An Expr or Form
        mapping: A dict with from:to replacements to perform

    Returns:
        The expression with replacements performed
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.items())

    # Workaround for problem with delayed derivative evaluation
    # The problem is that J = derivative(f(g, h), g) does not evaluate immediately
    # So if we subsequently do replace(J, {g: h}) we end up with an expression:
    # derivative(f(h, h), h)
    # rather than what were were probably thinking of:
    # replace(derivative(f(g, h), g), {g: h})
    #
    # To fix this would require one to expand derivatives early (which
    # is not attractive), or make replace lazy too.
    if has_exact_type(e, CoefficientDerivative):
        # Hack to avoid circular dependencies
        from ufl.algorithms.ad import expand_derivatives
        e = expand_derivatives(e)

    return map_integrand_dags(Replacer(mapping2), e)
