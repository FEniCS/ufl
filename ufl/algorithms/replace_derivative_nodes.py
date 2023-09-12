# -*- coding: utf-8 -*-
"""Algorithm for replacing derivative nodes in a BaseForm or Expr"""

import ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.constantvalue import as_ufl


class DerivativeNodeReplacer(MultiFunction):
    """Replace derivative nodes with new derivative nodes"""

    def __init__(self, mapping, **derivative_kwargs):
        super().__init__()
        self.mapping = mapping
        self.der_kwargs = derivative_kwargs

    expr = MultiFunction.reuse_if_untouched

    def coefficient_derivative(self, cd, o, coefficients, arguments, coefficient_derivatives):
        der_kwargs = self.der_kwargs
        new_coefficients = tuple(self.mapping[c] if c in self.mapping.keys() else c for c in coefficients.ufl_operands)

        # Ensure type compatibility for arguments!
        if 'argument' not in der_kwargs.keys():
            # Argument's number/part can be retrieved from the former coefficient derivative.
            arguments = tuple(type(a)(c.ufl_function_space(), a.number(), a.part())
                              for c, a in zip(new_coefficients, arguments.ufl_operands))
            der_kwargs.update({'argument': arguments})

        return ufl.derivative(o, new_coefficients, **der_kwargs)


def replace_derivative_nodes(expr, mapping, **derivative_kwargs):
    """Replaces derivative nodes, i.e. replaces the variable with respect to which the derivative is taken.
    This is called during apply_derivatives to treat delayed derivatives.

    Example: Let u be a Coefficient, N an ExternalOperator independent of u (i.e. N's operands don't depend on u),
             and let uhat and Nhat be Arguments.

        F = u ** 2 * N * dx
        dFdu = derivative(F, u, uhat)
        dFdN = replace_derivative_nodes(dFdu, {u: N}, argument=Nhat)

        Then, by subsequently expanding the derivatives we have:

        dFdu -> 2 * u * uhat * N * dx
        dFdN -> u ** 2 * Nhat * dx

    @param e:
        An Expr or BaseForm.
    @param mapping:
        A dict with from:to replacements to perform.
    @param derivative_kwargs:
        A dict containing the keyword arguments for derivative (i.e. `argument` and `coefficient_derivatives`).
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.items())
    return map_integrand_dags(DerivativeNodeReplacer(mapping2, **derivative_kwargs), expr)
