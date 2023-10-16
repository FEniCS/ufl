# Copyright (C) 2008 Anders Logg and Kristian B. Oelgaard
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
# This simple example illustrates how forms can be defined on different sub domains.
# It is supported for all three integral types.
from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, ds, dS, dx, tetrahedron
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pullback, H1)
domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pullback, H1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)

a = v * u * dx(0) + 10.0 * v * u * dx(1) + v * u * ds(0) + 2.0 * v * u * ds(1)\
    + v('+') * u('+') * dS(0) + 4.3 * v('+') * u('+') * dS(1)
