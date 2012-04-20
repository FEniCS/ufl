"""Front-end for AD routines."""

# Copyright (C) 2008-2012 Martin Sandve Alnes
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
#
# First added:  2008-12-28
# Last changed: 2012-04-12

from ufl.log import debug, error
from ufl.assertions import ufl_assert
from ufl.classes import Terminal, Derivative

from ufl.algorithms.transformer import transform_integrands, Transformer
from ufl.algorithms.expand_compounds import expand_compounds
from ufl.algorithms.reverse_ad import reverse_ad
from ufl.algorithms.forward_ad import forward_ad, apply_nested_forward_ad

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
        #print 'T apply_ad', e
        return e
    else:
        #print 'O apply_ad', e
        ops1 = e.operands()
        ops2 = tuple(apply_ad(o, ad_routine) for o in ops1)
        if not (ops1 == ops2):
            e = e.reconstruct(*ops2)
        if isinstance(e, Derivative):
            #print 'apply_ad calling ad_routine', e
            e = ad_routine(e)
        return e

def expand_derivatives1(form, dim=None,
                       apply_expand_compounds_before=True,
                       apply_expand_compounds_after=False,
                       use_alternative_wrapper_algorithm=False):
    """Expand all derivatives of expr.

    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or CoefficientDerivative objects left, and SpatialDerivative
    objects have been propagated to Terminal nodes."""

    # Find geometric dimension. This is messy because of PyDOLFIN integration issues.
    cell = form.cell()
    gdim = None if (cell is None or cell.is_undefined()) else cell.geometric_dimension()
    if dim is None:
        dim = gdim
    if gdim is not None:
        ufl_assert(dim == gdim, "Expecting dim to match the geometric dimension, got dim=%r and gdim=%r." % (dim, gdim))

    # Wrapper for AD algorithm of choice. Currently only forward mode is working.
    # TODO: How to switch between forward and reverse mode?
    #       Can we pick the best algorithm in each context?
    #       Could also try a mixed implementation working on the computational graph.
    def ad_routine(e):
        #print 'ad_routine:', e
        return forward_ad(e, dim)
        #return reverse_ad(e, dim)

    aa = ADApplyer(ad_routine)
    def _expand_derivatives(expression):
        #print '_expand_derivatives:', expression
        # Expand compound expressions or not, in the future this
        # should be removed from here and applied on the outside.
        if apply_expand_compounds_before:
            expression = expand_compounds(expression, dim)
            #print 'after expand_compounds', expression

        # Pick high level algorithm to use (for testing alternative 2 before we decide)
        if use_alternative_wrapper_algorithm:
            expression = apply_ad(expression, ad_routine)
        else:
            expression = aa.visit(expression)

        # FIXME: Form compilers assume expand_compounds have been applied.
        #        This means quite a bit of work to handle all compounds
        #        through the entire jit chain. For now, just test if we
        #        can apply compounds afterwards, to focus on fixing issues
        #        in the AD algorithm for compounds. Since this is optional,
        #        alternative form compilers can then disable expand_compounds alltogether.
        if apply_expand_compounds_after:
            expression = expand_compounds(expression, dim)

        return expression

    # Apply chosen algorithm to all integrands
    return transform_integrands(form, _expand_derivatives)



def expand_derivatives2(form, dim=None,
                       apply_expand_compounds_before=True,
                       apply_expand_compounds_after=False,
                       use_alternative_wrapper_algorithm=False):
    """Expand all derivatives of expr.

    In the returned expression g which is mathematically
    equivalent to expr, there are no VariableDerivative
    or CoefficientDerivative objects left, and SpatialDerivative
    objects have been propagated to Terminal nodes."""

    # Find geometric dimension. This is messy because of PyDOLFIN integration issues.
    cell = form.cell()
    gdim = None if (cell is None or cell.is_undefined()) else cell.geometric_dimension()
    if dim is None:
        dim = gdim
    if gdim is not None:
        ufl_assert(dim == gdim, "Expecting dim to match the geometric dimension, got dim=%r and gdim=%r." % (dim, gdim))

    def _expand_derivatives(expression):
        #print '_expand_derivatives:', expression
        # Expand compound expressions or not, in the future this
        # should be removed from here and applied on the outside.
        if apply_expand_compounds_before:
            expression = expand_compounds(expression, dim)
            #print 'after expand_compounds', expression

        # Apply recursive forward mode AD
        expression = apply_nested_forward_ad(expression, dim)

        # FIXME: Form compilers assume expand_compounds have been applied.
        #        This means quite a bit of work to handle all compounds
        #        through the entire jit chain. For now, just test if we
        #        can apply compounds afterwards, to focus on fixing issues
        #        in the AD algorithm for compounds. Since this is optional,
        #        alternative form compilers can then disable expand_compounds alltogether.
        if apply_expand_compounds_after:
            expression = expand_compounds(expression, dim)

        return expression

    # Apply chosen algorithm to all integrands
    return transform_integrands(form, _expand_derivatives)


expand_derivatives = expand_derivatives2
