# Copyright (C) 2006-2007 Anders Logg
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
# The bilinear form a(v, u) and linear form L(v) for
# Poisson's equation with Neumann boundary conditions.
from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, ds, dx, grad, inner, triangle
from ufl.finiteelement import FiniteElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
f = Coefficient(space)
g = Coefficient(space)

a = inner(grad(v), grad(u)) * dx
L = inner(v, f) * dx + inner(v, g) * ds
