# -*- coding: utf-8 -*-
"""Algorithm for replacing terminals in an expression."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs and Anders Logg
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009-2010

from ufl.classes import (CoefficientDerivative, Expr)
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import reuse_if_untouched
from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.analysis import has_exact_type
from functools import singledispatch


@singledispatch
def _replace(o, self, *args):
    """Single-dispatch function to replace subexpressions in expression

    :arg o: UFL expression
    :arg self: Callback handler that holds the mapping

    """
    raise AssertionError("UFL node expected, not %s" % type(o))


@_replace.register(Expr)
def _replace_expr(o, self, *args):
    try:
        return self.mapping[o]
    except KeyError:
        return reuse_if_untouched(o, *args)


@_replace.register(CoefficientDerivative)
def _replace_cofficient_derivative(o, self):
    raise ValueError("Derivatives should be applied before executing replace.")


class ReplaceWrapper(object):
    """
    :arg function: a function with parameters (value, rec), where
                   ``rec`` is expected to be a function used for
                   recursive calls.
    :arg mapping: a dict that describes the mapping of subexpressions to be replaced
    :returns: a function with working recursion and access to the
    """
    def __init__(self, function, mapping):
        self.cache = {}
        self.function = function
        self.mapping = mapping

    def __call__(self, node, *args):
        return self.function(node, self, *args)


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

    return map_integrand_dags(ReplaceWrapper(_replace,  mapping2), e)
