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

from ufl.utils.py23 import as_native_strings
from ufl.log import error
from ufl.finiteelement import FiniteElement, VectorElement, TensorElement, \
    MixedElement, EnrichedElement, NodalEnrichedElement

__all__ = as_native_strings(['increase_order', 'tear'])


def increase_order(element):
    "Return element of same family, but a polynomial degree higher."
    return _increase_degree(element, +1)


def change_regularity(element, family):
    """
    For a given finite element, return the corresponding space
    specified by 'family'.
    """
    return _change_family(element, family)


def tear(element):
    "For a finite element, return the corresponding discontinuous element."
    return change_regularity(element, "DG")


def reconstruct_element(element, family, cell, degree):
    if isinstance(element, FiniteElement):
        return FiniteElement(family, cell, degree)
    elif isinstance(element, VectorElement):
        return VectorElement(family, cell, degree, dim=element.value_shape()[0])
    elif isinstance(element, TensorElement):
        return TensorElement(family, cell, degree, shape=element.value_shape())
    else:
        error("Element reconstruction is only done to stay compatible"
              " with hacks in DOLFIN. Not expecting a %s" % repr(element))


def _increase_degree(element, degree_rise):
    if isinstance(element, (FiniteElement, VectorElement, TensorElement)):
        return reconstruct_element(element, element.family(), element.cell(),
                                   element.degree() + degree_rise)
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


def _change_family(element, family):
    if isinstance(element, (FiniteElement, VectorElement, TensorElement)):
        return reconstruct_element(element, family, element.cell(), element.degree())
    elif isinstance(element, MixedElement):
        return MixedElement([_change_family(e, family)
                             for e in element.sub_elements()])
    elif isinstance(element, EnrichedElement):
        return EnrichedElement([_change_family(e, family)
                                for e in element.sub_elements()])
    elif isinstance(element, NodalEnrichedElement):
        return NodalEnrichedElement([_change_family(e, family)
                                     for e in element.sub_elements()])
    else:
        error("Element reconstruction is only done to stay compatible"
              " with hacks in DOLFIN. Not expecting a %s" % repr(element))
