# Copyright (C) 2012 Marie E. Rognes
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
# First added:  2011-01-17
# Last changed: 2011-06-02

from ufl.assertions import ufl_assert
from ufl.finiteelement import FiniteElement, MixedElement

def change_regularity(element, family):
    """
    For a given finite element, return the corresponding space
    specified by 'family'.
    """

    n = element.num_sub_elements()
    if n > 0:
        subs = element.sub_elements()
        return MixedElement([change_regularity(subs[i], family)
                             for i in range(n)])
    shape = element.value_shape()
    if not shape:
        return FiniteElement(family, element.domain(), element.degree())

    ufl_assert(len(shape) == 1, "TODO: Update this code to handle tensor elements.")
    return MixedElement([FiniteElement(family, element.domain(i), element.degree(i))
                               for i in range(shape[0])])

def tear(V):
    "For a finite element, return the corresponding discontinuous element."
    W = change_regularity(V, "DG")
    return W

def increase_order(element):
    "Return element of same family, but a polynomial degree higher."
    ufl_assert(len(element.value_shape()) <= 1, "TODO: Update this code to handle tensor elements.")

    n = element.num_sub_elements()
    if n > 0:
        subs = element.sub_elements()
        return MixedElement([increase_order(subs[i]) for i in range(n)])

    if element.family() == "Real":
        return element

    return FiniteElement(element.family(), element.domain(), element.degree()+1)
