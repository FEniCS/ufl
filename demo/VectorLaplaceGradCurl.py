# Copyright (C) 2007 Marie Rognes
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
# The bilinear form a(v, u) and linear form L(v) for the Hodge Laplace
# problem using 0- and 1-forms. Intended to demonstrate use of Nedelec
# elements.
from ufl import Coefficient, FunctionSpace, Mesh, TestFunctions, TrialFunctions, curl, dx, grad, inner, tetrahedron
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pull_back import covariant_piola, identity_pull_back
from ufl.sobolevspace import H1, HCurl


def HodgeLaplaceGradCurl(space, fspace):
    tau, v = TestFunctions(space)
    sigma, u = TrialFunctions(space)
    f = Coefficient(fspace)

    a = (inner(tau, sigma) - inner(grad(tau), u) + inner(v, grad(sigma)) + inner(curl(v), curl(u))) * dx
    L = inner(v, f) * dx

    return a, L


cell = tetrahedron
order = 1

GRAD = FiniteElement("Lagrange", cell, order, (), (), identity_pull_back, H1)
CURL = FiniteElement("N1curl", cell, order, (3, ), (3, ), covariant_piola, HCurl)

VectorLagrange = FiniteElement("Lagrange", cell, order + 1, (3, ), (3, ), identity_pull_back, H1)

domain = Mesh(FiniteElement("Lagrange", cell, 1, (3, ), (3, ), identity_pull_back, H1))
space = FunctionSpace(domain, MixedElement([GRAD, CURL]))
fspace = FunctionSpace(domain, VectorLagrange)

a, L = HodgeLaplaceGradCurl(space, fspace)
