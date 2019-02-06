# -*- coding: utf-8 -*-
"""This module provides helper functions to
  - FFC/DOLFIN adaptive chain,
  - UFL algorithms taking care of underspecified DOLFIN expressions."""

# Copyright (C) 2012 Marie E. Rognes, 2015 Jan Blechta
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

from ufl.finiteelement import (EnrichedElement, FiniteElement, MixedElement,
                               NodalEnrichedElement, TensorElement,
                               VectorElement)
from ufl.log import error
from ufl.utils.str import as_native_strings

__all__ = as_native_strings(['increase_order', 'tear'])


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
