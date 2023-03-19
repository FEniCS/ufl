# -*- coding: utf-8 -*-
"""Algorithm for replacing terminals in an expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

from ufl.core.external_operator import ExternalOperator
from ufl.classes import CoefficientDerivative, Interp, Form
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.analysis import has_exact_type


class Replacer(MultiFunction):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

        # One can replace Coarguments by 1-Forms
        def get_shape(x):
            if isinstance(x, Form):
                return x.arguments()[0].ufl_shape
            return x.ufl_shape

        if not all(get_shape(k) == get_shape(v) for k, v in mapping.items()):
            raise ValueError("Replacement expressions must have the same shape as what they replace.")

    def ufl_type(self, o, *args):
        try:
            return self.mapping[o]
        except KeyError:
            return self.reuse_if_untouched(o, *args)

    """
    def external_operator(self, o):
        try:
            o = self.mapping[o]
            coeff = o.result_coefficient(unpack_reference=False)
        except KeyError:
            coeff = replace(o.result_coefficient(unpack_reference=False), self.mapping)
        except AttributeError:
            # ExternalOperator is replaced by something that is not an ExternalOperator
            return o
        new_ops = tuple(replace(op, self.mapping) for op in o.ufl_operands)
        # Fix this
        if type(new_ops[0]).__name__ == 'Coefficient' and type(o.ufl_operands[0]).__name__ == 'Function':
            new_ops = o.ufl_operands

        # Does not need to use replace on the 0-slot argument of external operators (v*) since
        # this can only be a Coargument or a Cofunction, so we directly check into the mapping.
        # Also, replace is built to apply on Expr, Coargument is not an Expr.
        new_args = tuple(replace(arg, self.mapping) if not isinstance(arg, Coargument) else self.mapping.get(arg, arg)
                         for arg in o.argument_slots())
        return o._ufl_expr_reconstruct_(*new_ops, result_coefficient=coeff, argument_slots=new_args)
    """

    def external_operator(self, o):
        o = self.mapping.get(o) or o
        if isinstance(o, ExternalOperator):
            new_ops = tuple(replace(op, self.mapping) for op in o.ufl_operands)
            new_args = tuple(replace(arg, self.mapping) for arg in o.argument_slots())
            return o._ufl_expr_reconstruct_(*new_ops, argument_slots=new_args)
        return o

    def interp(self, o):
        o = self.mapping.get(o) or o
        if isinstance(o, Interp):
            new_args = tuple(replace(arg, self.mapping) for arg in o.argument_slots())
            return o._ufl_expr_reconstruct_(*reversed(new_args))
        return o

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
