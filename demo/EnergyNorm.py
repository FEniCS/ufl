# Copyright (C) 2005-2007 Anders Logg
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
# This example demonstrates how to define a functional, here
# the energy norm (squared) for a reaction-diffusion problem.
from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, dot, dx, grad, tetrahedron

element = LagrangeElement(tetrahedron, 1)
domain = Mesh(LagrangeElement(tetrahedron, 1, (3,)))
space = FunctionSpace(domain, element)

v = Coefficient(space)
a = (v * v + dot(grad(v), grad(v))) * dx
