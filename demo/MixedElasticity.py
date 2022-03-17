# Copyright (C) 2008-2010 Marie Rognes
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
# First added:  2008-10-03
# Last changed: 2011-07-22
from ufl import (MixedElement, TestFunctions, TrialFunctions, VectorElement,
                 as_vector, div, dot, dx, inner, skew, tetrahedron, tr)


def skw(tau):
    """Define vectorized skew operator"""
    sk = 2 * skew(tau)
    return as_vector((sk[0, 1], sk[0, 2], sk[1, 2]))


cell = tetrahedron
n = 3

# Finite element exterior calculus syntax
r = 1
S = VectorElement("P Lambda", cell, r, form_degree=n - 1)
V = VectorElement("P Lambda", cell, r - 1, form_degree=n)
Q = VectorElement("P Lambda", cell, r - 1, form_degree=n)

# Alternative syntax:
# S = VectorElement("BDM", cell, r)
# V = VectorElement("Discontinuous Lagrange", cell, r-1)
# Q = VectorElement("Discontinuous Lagrange", cell, r-1)

W = MixedElement(S, V, Q)

(sigma, u, gamma) = TrialFunctions(W)
(tau, v, eta) = TestFunctions(W)

a = (inner(sigma, tau) - tr(sigma) * tr(tau) +
     dot(div(tau), u) - dot(div(sigma), v) +
     inner(skw(tau), gamma) + inner(skw(sigma), eta)) * dx
