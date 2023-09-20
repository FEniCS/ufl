# Copyright (C) 2005-2009 Anders Logg
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
# The bilinear form a(v, u1) and linear form L(v) for
# one backward Euler step with the heat equation.
#
from ufl import (Coefficient, Constant, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction, VectorElement,
                 dot, dx, grad, triangle)
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

cell = triangle
element = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
space = FunctionSpace(domain, element)

v = TestFunction(space)  # Test function
u1 = TrialFunction(space)  # Value at t_n
u0 = Coefficient(space)      # Value at t_n-1
c = Coefficient(space)      # Heat conductivity
f = Coefficient(space)      # Heat source
k = Constant(domain)         # Time step

a = v * u1 * dx + k * c * dot(grad(v), grad(u1)) * dx
L = v * u0 * dx + k * v * f * dx
