# Copyright (C) 2004-2007 Anders Logg
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
# Modified by Martin Sandve Alnes
#
# Last changed: 2009-03-02
#
# The bilinear form for the nonlinear term in the
# Navier-Stokes equations with fixed convective velocity.
from utils import LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dot,
    dx,
    grad,
    tetrahedron,
)

cell = tetrahedron
element = LagrangeElement(cell, 1, (3,))
domain = Mesh(LagrangeElement(cell, 1, (3,)))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
w = Coefficient(space)

Du = grad(u)
a = dot(dot(w, Du), v) * dx
