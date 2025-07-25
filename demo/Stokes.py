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
# Modified by: Martin Sandve Alnes (2009)
# Date: 2009-03-02
#
# The bilinear form a(v, u) and Linear form L(v) for the Stokes
# equations using a mixed formulation (Taylor-Hood elements).
from utils import LagrangeElement, MixedElement

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
    triangle,
)

cell = triangle
P2 = LagrangeElement(cell, 2, (2,))
P1 = LagrangeElement(cell, 1)
TH = MixedElement([P2, P1])
domain = Mesh(LagrangeElement(cell, 1, (2,)))
space = FunctionSpace(domain, TH)
p2_space = FunctionSpace(domain, P2)

(v, q) = TestFunctions(space)
(u, p) = TrialFunctions(space)

f = Coefficient(p2_space)

a = (inner(grad(v), grad(u)) - div(v) * p + q * div(u)) * dx
L = dot(v, f) * dx
