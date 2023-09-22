#
# Author: Marie Rognes
# Modified by: Martin Sandve Alnes
# Date: 2009-02-12
#
from ufl import FacetNormal, FunctionSpace, Mesh, TestFunctions, TrialFunctions, div, dot, ds, dx, tetrahedron
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pull_back import contravariant_piola, identity_pull_back
from ufl.sobolevspace import H1, HDiv

cell = tetrahedron
RT = FiniteElement("Raviart-Thomas", cell, 1, (3, ), (3, ), contravariant_piola, HDiv)
DG = FiniteElement("DG", cell, 0, (), (), identity_pull_back, H1)
MX = MixedElement([RT, DG])
domain = Mesh(FiniteElement("Lagrange", cell, 1, (3, ), (3, ), identity_pull_back, H1))
space = FunctionSpace(domain, MX)

(u, p) = TrialFunctions(space)
(v, q) = TestFunctions(space)

n = FacetNormal(domain)

a0 = (dot(u, v) + div(u) * q + div(v) * p) * dx
a1 = (dot(u, v) + div(u) * q + div(v) * p) * dx - p * dot(v, n) * ds

forms = [a0, a1]
