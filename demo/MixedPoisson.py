# Copyright (C) 2006-2009 Anders Logg and Marie Rognes
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
# Modified by Martin Sandve Alnes, 2009
#
# Last changed: 2009-01-12
#
# The bilinear form a(v, u) and linear form L(v) for
# a mixed formulation of Poisson's equation with BDM
# (Brezzi-Douglas-Marini) elements.
#
from ufl import Coefficient, FunctionSpace, Mesh, TestFunctions, TrialFunctions, div, dot, dx, triangle
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pullback import contravariant_piola, identity_pullback
from ufl.sobolevspace import H1, HDiv

cell = triangle
BDM1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1, (2, ), contravariant_piola, HDiv)
DG0 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, H1)

element = MixedElement([BDM1, DG0])
domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1))
space = FunctionSpace(domain, element)
dg0_space = FunctionSpace(domain, DG0)

(tau, w) = TestFunctions(space)
(sigma, u) = TrialFunctions(space)

f = Coefficient(dg0_space)

a = (dot(tau, sigma) - div(tau) * u + w * div(sigma)) * dx
L = w * f * dx
