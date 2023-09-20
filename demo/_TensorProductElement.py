# Copyright (C) 2012 Marie E. Rognes (meg@simula.no)
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
# First added:  2012-08-16
# Last changed: 2012-08-16
from ufl import (FunctionSpace, Mesh, TensorProductElement, TestFunction, TrialFunction, dx, interval, tetrahedron,
                 triangle)
from ufl.sobolevspace import H1, L2

V0 = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
V1 = FiniteElement("DG", interval, 0, (), (), "identity", L2)
V2 = FiniteElement("DG", tetrahedron, 0, (), (), "identity", L2)

V = TensorProductElement(V0, V1, V2)

c0 = FiniteElement("CG", triangle, 1)
c1 = FiniteElement("CG", interval, 1)
c2 = FiniteElement("CG", tetrahedron, 1)
domain = Mesh(TensorProductElement(c0, c1, c2))
space = FunctionSpace(domain, V)

u = TrialFunction(space)
v = TestFunction(space)

dxxx = dx * dx * dx
a = u * v * dxxx
