#
# Author: Anders Logg
# Modified by: Martin Sandve Alnes
# Date: 2009-01-12
#
from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, dx, grad, inner, tetrahedron
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1)
domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)


def epsilon(v):
    Dv = grad(v)
    return 0.5 * (Dv + Dv.T)


a = inner(epsilon(v), epsilon(u)) * dx
