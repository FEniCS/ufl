# Copyright (C) 2008 Kristian B. Oelgaard
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
# First added:  2008-03-31
# Last changed: 2008-03-31
#
# The linearised bilinear form a(u,v) and linear form L(v) for
# the nonlinear equation - div (1+u) grad u = f (non-linear Poisson)
from ufl import Coefficient, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction, dot, dx, grad, i, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 2, (), (), "identity", H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
space = FunctionSpace(domain, element)

QE = FiniteElement("Quadrature", triangle, 2, (), (), "identity", H1)
sig = FiniteElement("Quadrature", triangle, 1, (2, ), (2, ), "identity", H1)

qe_space = FunctionSpace(domain, QE)
sig_space = FunctionSpace(domain, sig)

v = TestFunction(space)
u = TrialFunction(space)
u0 = Coefficient(space)
C = Coefficient(qe_space)
sig0 = Coefficient(sig_space)
f = Coefficient(space)

a = v.dx(i) * C * u.dx(i) * dx(metadata={"quadrature_degree": 2}) + v.dx(i) * 2 * u0 * u * u0.dx(i) * dx
L = v * f * dx - dot(grad(v), sig0) * dx(metadata={"quadrature_degree": 1})
