"""This module provides the necessary tools to strip away and reattach coordinate derivatives.

This is used in compute_form_data.
"""

# Copyright (C) 2018 Florian Wechsung
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.differentiation import CoordinateDerivative
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dags
from ufl.classes import Integral


class CoordinateDerivativeIsOutermostChecker(MultiFunction):
    """Traverses the tree to make sure that CoordinateDerivatives are only on the outside.

    The visitor returns False as long as no CoordinateDerivative has been seen.
    """

    def multi_index(self, o):
        """Apply to multi_index."""
        return False

    def terminal(self, o):
        """Apply to terminal."""
        return False

    def expr(self, o, *operands):
        """Apply to expr.

        If we have already seen a CoordinateDerivative, then no other
        expressions apart from more CoordinateDerivatives are allowed to wrap
        around it.
        """
        if any(operands):
            raise ValueError("CoordinateDerivative(s) must be outermost")
        return False

    def coordinate_derivative(self, o, expr, *_):
        """Apply to coordinate derivative."""
        return True


def strip_coordinate_derivatives(integrals):
    """Strip coordinate derivatives."""
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

        def take_top_coordinate_derivatives(o):
            """Grab all coordinate derivatives and store them, so that we can apply them later again."""
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
    """Attach coordinate derivatives."""
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
