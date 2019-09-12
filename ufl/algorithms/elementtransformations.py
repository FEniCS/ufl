# -*- coding: utf-8 -*-
"""This module provides helper functions to
  - FFC/DOLFIN adaptive chain,
  - UFL algorithms taking care of underspecified DOLFIN expressions."""

# Copyright (C) 2012 Marie E. Rognes, 2015 Jan Blechta
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.log import error
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement, \
    MixedElement, EnrichedElement, NodalEnrichedElement

__all__ = ['increase_order', 'tear']


def increase_order(element):
    "Return element of same family, but a polynomial degree higher."
    return _increase_degree(element, +1)


def change_regularity(element, family):
    """
    For a given finite element, return the corresponding space
    specified by 'family'.
    """
    return element.reconstruct(family=family)


def tear(element):
    "For a finite element, return the corresponding discontinuous element."
    return change_regularity(element, "DG")


def _increase_degree(element, degree_rise):
    if isinstance(element, (FiniteElement, VectorElement, TensorElement)):
        # Can't increase degree for reals
        if element.family() == "Real":
            return element
        return element.reconstruct(degree=(element.degree() + degree_rise))
    elif isinstance(element, MixedElement):
        return MixedElement([_increase_degree(e, degree_rise)
                             for e in element.sub_elements()])
    elif isinstance(element, EnrichedElement):
        return EnrichedElement([_increase_degree(e, degree_rise)
                                for e in element.sub_elements()])
    elif isinstance(element, NodalEnrichedElement):
        return NodalEnrichedElement([_increase_degree(e, degree_rise)
                                     for e in element.sub_elements()])
    else:
        error("Element reconstruction is only done to stay compatible"
              " with hacks in DOLFIN. Not expecting a %s" % repr(element))
