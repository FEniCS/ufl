# Copyright (C) 2005-2009 Anders Logg and Harish Narayanan
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
# The bilinear form a(v, u) and Linear form L(v) for the Stokes
# equations using a mixed formulation (Taylor-Hood elements) in
# combination with the lhs() and rhs() operators to extract the
# bilinear and linear forms from an expression F = 0.
from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunctions,
    TrialFunctions,
    div,
    dot,
    dx,
    grad,
    inner,
    lhs,
    rhs,
    triangle,
)
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

cell = triangle
P2 = FiniteElement("Lagrange", cell, 2, (2,), identity_pullback, H1)
P1 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
TH = MixedElement([P2, P1])
domain = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1))
space = FunctionSpace(domain, TH)
p2_space = FunctionSpace(domain, P2)

(v, q) = TestFunctions(space)
(u, p) = TrialFunctions(space)

f = Coefficient(p2_space)

F = (inner(grad(v), grad(u)) - div(v) * p + q * div(u)) * dx - dot(v, f) * dx
a = lhs(F)
L = rhs(F)
