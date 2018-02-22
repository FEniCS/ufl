# -*- coding: utf-8 -*-
"""Algorithm for replacing terminals in an expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009-2010

from ufl.log import error
from ufl.classes import CoefficientDerivative
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.analysis import has_exact_type


class Replacer(MultiFunction):
    def __init__(self, mapping):
        MultiFunction.__init__(self)
        self._mapping = mapping
        if not all(k._ufl_is_terminal_ for k in mapping.keys()):
            error("This implementation can only replace Terminal objects.")
        if not all(k.ufl_shape == v.ufl_shape for k, v in mapping.items()):
            error("Replacement expressions must have the same shape as what they replace.")

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, o):
        e = self._mapping.get(o)
        if e is None:
            return o
        else:
            return e

    def coefficient_derivative(self, o):
        error("Derivatives should be applied before executing replace.")


def replace(e, mapping):
    """Replace terminal objects in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.items())

    # Workaround for problem with delayed derivative evaluation
    if has_exact_type(e, CoefficientDerivative):
        # Hack to avoid circular dependencies
        from ufl.algorithms.ad import expand_derivatives
        e = expand_derivatives(e)

    return map_integrand_dags(Replacer(mapping2), e)
