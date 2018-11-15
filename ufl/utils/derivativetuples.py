# -*- coding: utf-8 -*-
"This module contains a collection of utilities for representing partial derivatives as integer tuples."

# Copyright (C) 2013-2016 Martin Sandve Alnæs and Anders Logg
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

import itertools


def derivative_counts_to_listing(derivative_counts):
    """Convert a derivative count tuple to a derivative listing tuple.

    The derivative d^3 / dy^2 dz is represented
    in counting form as (0, 2, 1) meaning (dx^0, dy^2, dz^1)
    and in listing form as (1, 1, 2) meaning (dy, dy, dz).
    """
    derivatives = []  # = 1
    for i, d in enumerate(derivative_counts):
        derivatives.extend((i,) * d)  # *= d/dx_i^d
    return tuple(derivatives)


def derivative_listing_to_counts(derivatives, gdim):
    """Convert a derivative listing tuple to a derivative count tuple.

    The derivative d^3 / dy^2 dz is represented
    in counting form as (0, 2, 1) meaning (dx^0, dy^2, dz^1)
    and in listing form as (1, 1, 2) meaning (dy, dy, dz).
    """
    derivative_counts = [0] * gdim
    for d in derivatives:
        derivative_counts[d] += 1
    return tuple(derivative_counts)


def compute_derivative_tuples(n, gdim):
    """Compute the list of all derivative tuples for derivatives of
    given total order n and given geometric dimension gdim. This
    function returns two lists. The first is a list of tuples, where
    each tuple of length n specifies the coordinate directions of the
    n derivatives. The second is a corresponding list of tuples, where
    each tuple of length gdim specifies the number of derivatives in
    each direction. Both lists have length gdim^n and are ordered as
    expected by the UFC function tabulate_basis_derivatives.

    Example: If n = 2 and gdim = 3, then the nice tuples are

      (0, 0)  <-->  (2, 0, 0)  <-->  d^2/dxdx
      (0, 1)  <-->  (1, 1, 0)  <-->  d^2/dxdy
      (0, 2)  <-->  (1, 0, 1)  <-->  d^2/dxdz
      (1, 0)  <-->  (1, 1, 0)  <-->  d^2/dydx
      (1, 1)  <-->  (0, 2, 0)  <-->  d^2/dydy
      (1, 2)  <-->  (0, 1, 1)  <-->  d^2/dydz
      (2, 0)  <-->  (1, 0, 1)  <-->  d^2/dzdx
      (2, 1)  <-->  (0, 1, 1)  <-->  d^2/dzdy
      (2, 2)  <-->  (0, 0, 2)  <-->  d^2/dzdz
    """

    # Create list of derivatives (note that we have d^n derivatives)
    deriv_tuples = [d for d in itertools.product(*(n * [range(0, gdim)]))]

    # Translate from list of derivative tuples to list of tuples
    # expressing the number of derivatives in each dimension...
    _deriv_tuples = [tuple(len([_d for _d in d if _d == i]) for i in range(gdim))
                     for d in deriv_tuples]

    return deriv_tuples, _deriv_tuples
