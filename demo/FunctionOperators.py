# Copyright (C) 2007 Anders Logg
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
# Test form for operators on Coefficients.
from ufl import (Coefficient, TestFunction, TrialFunction, dot, dx, grad,
                 max_value, sqrt, triangle)
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

v = TestFunction(element)
u = TrialFunction(element)
f = Coefficient(element)
g = Coefficient(element)

a = sqrt(1 / max_value(1 / f, -1 / f)) * sqrt(g) * dot(grad(v), grad(u)) * dx + v * u * sqrt(f * g) * g * dx
