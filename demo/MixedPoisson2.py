#
# Author: Marie Rognes
# Modified by: Martin Sandve Alnes
# Date: 2009-02-12
#
from ufl import (FacetNormal, TestFunctions, TrialFunctions, div, dot, ds, dx,
                 tetrahedron)
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.sobolevspace import H1, HDiv

cell = tetrahedron
RT = FiniteElement("Raviart-Thomas", cell, 1, (3, ), (3, ), "contravariant Piola", HDiv)
DG = FiniteElement("DG", cell, 0, (), (), "identity", H1)
MX = MixedElement([RT, DG])

(u, p) = TrialFunctions(MX)
(v, q) = TestFunctions(MX)

n = FacetNormal(cell)

a0 = (dot(u, v) + div(u) * q + div(v) * p) * dx
a1 = (dot(u, v) + div(u) * q + div(v) * p) * dx - p * dot(v, n) * ds

forms = [a0, a1]
