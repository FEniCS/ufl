"""Algorithm for replacing terminals in an expression."""

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
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
#
# First added:  2008-05-07
# Last changed: 2012-04-12

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.classes import Terminal, CoefficientDerivative
from ufl.constantvalue import as_ufl
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.algorithms.analysis import extract_type

class Replacer(ReuseTransformer):
    def __init__(self, mapping):
        ReuseTransformer.__init__(self)
        self._mapping = mapping
        ufl_assert(all(isinstance(k, Terminal) for k in mapping.keys()), \
            "This implementation can only replace Terminal objects.")

    def terminal(self, o):
        e = self._mapping.get(o)
        return o if e is None else e

    def coefficient_derivative(self, o):
        error("Coefficient derivatives should be expanded before applying replace.")

def replace(e, mapping):
    """Replace terminal objects in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    mapping2 = dict((k, as_ufl(v)) for (k,v) in mapping.iteritems()) # TODO: Should this be sorted?

    # Workaround for problem with delayed derivative evaluation
    if extract_type(e, CoefficientDerivative):
        # Hack to avoid circular dependencies
        from ufl.algorithms.ad import expand_derivatives
        e = expand_derivatives(e)

    return apply_transformer(e, Replacer(mapping2))
