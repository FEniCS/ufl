#
# Author: Marie Rognes
# Modified by: Martin Sandve Alnes
# Date: 2009-02-12
#
from ufl import (FacetNormal, FiniteElement, TestFunctions, TrialFunctions,
                 div, dot, ds, dx, tetrahedron)

cell = tetrahedron
RT = FiniteElement("Raviart-Thomas", cell, 1)
DG = FiniteElement("DG", cell, 0)
MX = RT * DG

(u, p) = TrialFunctions(MX)
(v, q) = TestFunctions(MX)

n = FacetNormal(cell)

a0 = (dot(u, v) + div(u) * q + div(v) * p) * dx
a1 = (dot(u, v) + div(u) * q + div(v) * p) * dx - p * dot(v, n) * ds

forms = [a0, a1]
