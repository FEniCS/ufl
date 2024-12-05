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
# Modified by: Martin Sandve Alnes, 2009
#
# Last changed: 2009-03-02
#
# The bilinear form a(v, u) and linear form L(v) for
# Poisson's equation in system form (vector-valued).
from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dot,
    dx,
    grad,
    inner,
    triangle,
)
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

cell = triangle
element = FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1)
domain = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
f = Coefficient(space)

a = inner(grad(v), grad(u)) * dx
L = dot(v, f) * dx
