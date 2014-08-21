"""Front-end for AD routines."""

# Copyright (C) 2008-2014 Martin Sandve Alnes
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
# Modified by Anders Logg, 2009.

from ufl.log import debug, error
from ufl.assertions import ufl_assert
from ufl.classes import Terminal, Derivative
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.expand_compounds import expand_compounds, expand_compounds_postdiff
from ufl.algorithms.forward_ad import apply_nested_forward_ad

def expand_derivatives(form, dim=None,
                       apply_expand_compounds_before=True,
                       apply_expand_compounds_after=False,
                       use_alternative_wrapper_algorithm=False):
    """Expand all derivatives of expr.

    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or CoefficientDerivative objects left, and Grad
    objects have been propagated to Terminal nodes.

    Note: Passing dim is now unnecessary, and the argument is ignored.
    """

    def _expand_derivatives(expression):
        #print '_expand_derivatives:', expression
        # Expand compound expressions or not, in the future this
        # should be removed from here and applied on the outside.
        if apply_expand_compounds_before:
            expression = expand_compounds(expression)
            #print 'after expand_compounds', expression

        # Apply recursive forward mode AD
        expression = apply_nested_forward_ad(expression)

        # FIXME: Form compilers assume expand_compounds have been applied.
        #        This means quite a bit of work to handle all compounds
        #        through the entire jit chain. For now, just test if we
        #        can apply compounds afterwards, to focus on fixing issues
        #        in the AD algorithm for compounds. Since this is optional,
        #        alternative form compilers can then disable expand_compounds alltogether.
        if apply_expand_compounds_after:
            # FIXME: Test expand_compounds_postdiff, it should make this algorithm viable for existing FFC code
            #expression = expand_compounds(expression)
            expression = expand_compounds_postdiff(expression)
        return expression

    # Apply chosen algorithm to all integrands
    return map_integrand_dags(_expand_derivatives, form)
