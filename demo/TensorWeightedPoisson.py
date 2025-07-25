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
# The bilinear form a(v, u) and linear form L(v) for
# tensor-weighted Poisson's equation.
from utils import FiniteElement, LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dx,
    grad,
    inner,
    triangle,
)
from ufl.pullback import identity_pullback
from ufl.sobolevspace import L2

P1 = LagrangeElement(triangle, 1)
P0 = FiniteElement("Discontinuous Lagrange", triangle, 0, (2, 2), identity_pullback, L2)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
p1_space = FunctionSpace(domain, P1)
p0_space = FunctionSpace(domain, P0)

v = TestFunction(p1_space)
u = TrialFunction(p1_space)
C = Coefficient(p0_space)

a = inner(grad(v), C * grad(u)) * dx
