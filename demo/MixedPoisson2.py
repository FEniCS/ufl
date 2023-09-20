#
# Author: Marie Rognes
# Modified by: Martin Sandve Alnes
# Date: 2009-02-12
#
from ufl import (FacetNormal, FiniteElement, FunctionSpace, Mesh, TestFunctions, TrialFunctions, VectorElement, div,
                 dot, ds, dx, tetrahedron)

cell = tetrahedron
RT = FiniteElement("Raviart-Thomas", cell, 1)
DG = FiniteElement("DG", cell, 0)
MX = RT * DG
domain = Mesh(VectorElement("Lagrange", cell, 1))
space = FunctionSpace(domain, MX)

(u, p) = TrialFunctions(space)
(v, q) = TestFunctions(space)

n = FacetNormal(domain)

a0 = (dot(u, v) + div(u) * q + div(v) * p) * dx
a1 = (dot(u, v) + div(u) * q + div(v) * p) * dx - p * dot(v, n) * ds

forms = [a0, a1]
