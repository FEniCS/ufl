# -*- coding: utf-8 -*-
"""This module provides the necessary tools to strip away and then reattach the
coordinate derivatives at the right time point in compute_form_data."""

# Copyright (C) 2018 Florian Wechsung
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

from ufl.algorithms.multifunction import MultiFunction
from ufl.classes import Integral
from ufl.corealg.map_dag import map_expr_dags
from ufl.differentiation import CoordinateDerivative
from ufl.log import error


class CoordinateDerivativeIsOutermostChecker(MultiFunction):

    """ Traverses the tree to make sure that CoordinateDerivatives are only on
    the outside. The visitor returns False as long as no CoordinateDerivative
    has been seen. """

    def multi_index(self, o):
        return False

    def terminal(self, o):
        return False

    def expr(self, o, *operands):
        """ If we have already seen a CoordinateDerivative, then no other
        expressions apart from more CoordinateDerivatives are allowed to wrap
        around it. """
        if any(operands):
            error("CoordinateDerivative(s) must be outermost")
        return False

    def coordinate_derivative(self, o, expr, *_):
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
        checker = CoordinateDerivativeIsOutermostChecker()
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
        error("Invalid type %s" % (integrals.__class__.__name__,))


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
        error("Invalid type %s" % (integral.__class__.__name__,))
