# -*- coding: utf-8 -*-
"""Algorithm for replacing terminals in an expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

from ufl.classes import CoefficientDerivative
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.analysis import has_exact_type


class Replacer(MultiFunction):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping
        if not all(k.ufl_shape == v.ufl_shape for k, v in mapping.items()):
            raise ValueError("Replacement expressions must have the same shape as what they replace.")

    def ufl_type(self, o, *args):
        try:
            return self.mapping[o]
        except KeyError:
            return self.reuse_if_untouched(o, *args)

    def external_operator(self, o):
        try:
            o = self.mapping[o]
            coeff = o.coefficient()
        except KeyError:
            coeff = replace(o.coefficient(), self.mapping)
        new_ops = tuple(replace(op, self.mapping) for op in o.ufl_operands)
        if type(new_ops[0]).__name__ == 'Coefficient' and type(o.ufl_operands[0]).__name__ == 'Function':
            new_ops = o.ufl_operands
        new_args = tuple((replace(arg, self.mapping), is_adjoint) for arg, is_adjoint in o.arguments())
        return o._ufl_expr_reconstruct_(*new_ops, coefficient=coeff, arguments=new_args)

    def coefficient_derivative(self, o):
        raise ValueError("Derivatives should be applied before executing replace.")


def replace(e, mapping):
    """Replace subexpressions in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
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
