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
from ufl import (FunctionSpace, Mesh, TestFunctions, TrialFunctions, as_vector, div, dot, dx, inner, skew, tetrahedron,
                 tr)
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pull_back import contravariant_piola, identity_pull_back
from ufl.sobolevspace import H1, L2, HDiv


def skw(tau):
    """Define vectorized skew operator"""
    sk = 2 * skew(tau)
    return as_vector((sk[0, 1], sk[0, 2], sk[1, 2]))


cell = tetrahedron
n = 3

# Finite element exterior calculus syntax
r = 1
S = FiniteElement("vector BDM", cell, r, (3, 3), (3, 3), contravariant_piola, HDiv)
V = FiniteElement("Discontinuous Lagrange", cell, r - 1, (3, ), (3, ), identity_pull_back, L2)
Q = FiniteElement("Discontinuous Lagrange", cell, r - 1, (3, ), (3, ), identity_pull_back, L2)

W = MixedElement([S, V, Q])

domain = Mesh(FiniteElement("Lagrange", cell, 1, (3, ), (3, ), identity_pull_back, H1))
space = FunctionSpace(domain, W)

(sigma, u, gamma) = TrialFunctions(space)
(tau, v, eta) = TestFunctions(space)

a = (
    inner(sigma, tau) - tr(sigma) * tr(tau) + dot(
        div(tau), u
    ) - dot(div(sigma), v) + inner(skw(tau), gamma) + inner(skw(sigma), eta)
) * dx
