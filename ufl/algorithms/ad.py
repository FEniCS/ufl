"""Front-end for AD routines."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL.  If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2009.
#
# First added:  2008-12-28
# Last changed: 2009-12-08

from itertools import izip
from ufl.log import debug, error
from ufl.assertions import ufl_assert
from ufl.classes import Terminal, Expr, Derivative, Tuple
from ufl.classes import SpatialDerivative, VariableDerivative, CoefficientDerivative
from ufl.classes import FiniteElement, TestFunction

from ufl.algorithms.analysis import extract_classes
from ufl.algorithms.transformations import transform_integrands, expand_compounds, Transformer
from ufl.algorithms.reverse_ad import reverse_ad
from ufl.algorithms.forward_ad import forward_ad

class ADApplyer(Transformer):
    def __init__(self, ad_routine):
        Transformer.__init__(self)
        self.ad_routine = ad_routine

    def terminal(self, e):
        return e

    def expr(self, e, *ops):
        e = Transformer.reuse_if_possible(self, e, *ops)
        if isinstance(e, Derivative):
            e = self.ad_routine(e)
        return e

def apply_ad(e, ad_routine):
    if isinstance(e, Terminal):
        return e
    ops1 = e.operands()
    ops = [apply_ad(o, ad_routine) for o in ops1]
    if not all(a is b for (a,b) in zip(ops, ops1)):
        e = e.reconstruct(*ops)
    if isinstance(e, Derivative):
        e = ad_routine(e)
    return e

def expand_derivatives(form):
    """Expand all derivatives of expr.

    NB! This functionality is not finished!

    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or CoefficientDerivative objects left, and SpatialDerivative
    objects have been propagated to Terminal nodes."""

    cell = form.cell()
    dim = None if cell is None else cell.geometric_dimension()

    def ad_routine(e):
        # TODO: How to switch between forward and reverse mode? Can we pick the
        #       best in each context? Want to try a mixed implementation on the
        #       graph.
        return forward_ad(e, dim)
        #return reverse_ad(e, dim)

    # TODO: This is probably faster, use after testing.
    #def _expand_derivatives(expression):
    #    expression = expand_compounds(expression, dim)
    #    return apply_ad(expression, ad_routine)

    aa = ADApplyer(ad_routine)
    def _expand_derivatives(expression):
        expression = expand_compounds(expression, dim)
        return aa.visit(expression)

    return transform_integrands(form, _expand_derivatives)
