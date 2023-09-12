# -*- coding: utf-8 -*-
"""This module provides the necessary tools to strip away and then reattach the
coordinate derivatives at the right time point in compute_form_data."""

# Copyright (C) 2018 Florian Wechsung
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.differentiation import CoordinateDerivative
from ufl.corealg.map_dag import map_expr_dags
from ufl.classes import (Integral, MultiIndex, Terminal, Expr)
from functools import singledispatch


@singledispatch
def coordinateDerivativeIsOutermost(o, *args):
    """ Traverses the tree to make sure that CoordinateDerivatives are only on
    the outside. The visitor returns False as long as no CoordinateDerivative
    has been seen. """
    raise AssertionError("UFL node expected, not %s" % type(o))


@coordinateDerivativeIsOutermost.register(MultiIndex)
@coordinateDerivativeIsOutermost.register(Terminal)
def coordinateDerivativeIsOutermost_False(o, *args):
    return False


@coordinateDerivativeIsOutermost.register(Expr)
def coordinateDerivativeIsOutermost_Expr(o, *args):
    """ If we have already seen a CoordinateDerivative, then no other
        expressions apart from more CoordinateDerivatives are allowed to wrap
        around it. """
    if any(args):
        raise ValueError("CoordinateDerivative(s) must be outermost")
    return False


@coordinateDerivativeIsOutermost.register(CoordinateDerivative)
def coordinateDerivativeIsOutermost_True(o, *args):
    return True


def strip_coordinate_derivatives(integrals):

    if isinstance(integrals, list):
        if len(integrals) == 0:
            return integrals, None
        stripped_integrals_and_cds = []
        for integral in integrals:
            (si, cd) = strip_coordinate_derivatives(integral)
            stripped_integrals_and_cds.append((si, cd))
        return stripped_integrals_and_cds

    elif isinstance(integrals, Integral):
        integral = integrals
        integrand = integral.integrand()
        checker = coordinateDerivativeIsOutermost
        map_expr_dags(checker, [integrand])
        coordinate_derivatives = []

        # grab all coordinate derivatives and store them, so that we can apply
        # them later again
        def take_top_coordinate_derivatives(o):
            o_ = o.ufl_operands
            if isinstance(o, CoordinateDerivative):
                coordinate_derivatives.append((o_[1], o_[2], o_[3]))
                return take_top_coordinate_derivatives(o_[0])
            else:
                return o

        newintegrand = take_top_coordinate_derivatives(integrand)
        return (integral.reconstruct(integrand=newintegrand), coordinate_derivatives)

    else:
        raise ValueError(f"Invalid type {integrals.__class__.__name__}")


def attach_coordinate_derivatives(integral, coordinate_derivatives):
    if coordinate_derivatives is None:
        return integral

    if isinstance(integral, Integral):
        integrand = integral.integrand()
        # apply the stored coordinate derivatives back onto the integrand
        for tup in reversed(coordinate_derivatives):
            integrand = CoordinateDerivative(integrand, tup[0], tup[1], tup[2])
        return integral.reconstruct(integrand=integrand)
    else:
        raise ValueError(f"Invalid type {integral.__class__.__name__}")
