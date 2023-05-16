#
# Author: Anders Logg
# Modified by: Martin Sandve Alnes
# Date: 2009-01-12
#
from ufl import TestFunction, TrialFunction, dx, grad, inner, tetrahedron
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", tetrahedron, 1, (3, ), (3, ), "identity", H1)

v = TestFunction(element)
u = TrialFunction(element)


def epsilon(v):
    Dv = grad(v)
    return 0.5 * (Dv + Dv.T)


a = inner(epsilon(v), epsilon(u)) * dx
