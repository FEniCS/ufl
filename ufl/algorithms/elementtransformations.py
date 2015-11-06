# -*- coding: utf-8 -*-
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

from ufl.algorithms.compute_form_data import _increase_degree, _change_family

__all__ = ['increase_order', 'tear']

def increase_order(element):
    "Return element of same family, but a polynomial degree higher."
    return _increase_degree(element, +1)

def change_regularity(element, family):
    """
    For a given finite element, return the corresponding space
    specified by 'family'.
    """
    return _change_element(element, family)

def tear(element):
    "For a finite element, return the corresponding discontinuous element."
    return change_regularity(element, "DG")
