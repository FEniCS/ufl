# -*- coding: utf-8 -*-
"This module contains a collection of utilities for mapping between multiindices and a flattened index space."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
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


def shape_to_strides(sh):
    "Return a tuple of strides given a shape tuple."
    n = len(sh)
    if not n:
        return ()
    strides = [None] * n
    strides[n - 1] = 1
    for i in range(n - 1, 0, -1):
        strides[i - 1] = strides[i] * sh[i]
    return tuple(strides)


def flatten_multiindex(ii, strides):
    "Return the flat index corresponding to the given multiindex."
    i = 0
    for c, s in zip(ii, strides):
        i += c * s
    return i


def unflatten_index(i, strides):
    "Return the multiindex corresponding to the given flat index."
    ii = []
    for s in strides:
        ii.append(i // s)
        i %= s
    return tuple(ii)
