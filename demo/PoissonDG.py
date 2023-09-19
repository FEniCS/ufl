# Copyright (C) 2006-2007 Kristiand Oelgaard and Anders Logg
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
# First added:  2006-12-05
# Last changed: 2007-07-15
#
# The bilinear form a(v, u) and linear form L(v) for
# Poisson's equation in a discontinuous Galerkin (DG)
# formulation.
from ufl import (Coefficient, Constant, FacetNormal, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction,
                 VectorElement, avg, dot, dS, ds, dx, grad, inner, jump, triangle)

cell = triangle
element = FiniteElement("Discontinuous Lagrange", cell, 1)
domain = Mesh(VectorElement("Lagrange", cell, 1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
f = Coefficient(space)

n = FacetNormal(domain)
h = Constant(domain)

gN = Coefficient(space)

alpha = 4.0
gamma = 8.0

a = inner(grad(v), grad(u)) * dx \
    - inner(avg(grad(v)), jump(u, n)) * dS \
    - inner(jump(v, n), avg(grad(u))) * dS \
    + alpha / h('+') * dot(jump(v, n), jump(u, n)) * dS \
    - inner(grad(v), u * n) * ds \
    - inner(v * n, grad(u)) * ds \
    + gamma / h * v * u * ds

L = v * f * dx + v * gN * ds
