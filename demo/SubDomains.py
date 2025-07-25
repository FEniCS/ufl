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
from utils import LagrangeElement

from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, dS, ds, dx, tetrahedron

element = LagrangeElement(tetrahedron, 1)
domain = Mesh(LagrangeElement(tetrahedron, 1, (3,)))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)

a = (
    v * u * dx(0)
    + 10.0 * v * u * dx(1)
    + v * u * ds(0)
    + 2.0 * v * u * ds(1)
    + v("+") * u("+") * dS(0)
    + 4.3 * v("+") * u("+") * dS(1)
)
