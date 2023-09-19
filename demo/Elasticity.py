#
# Author: Anders Logg
# Modified by: Martin Sandve Alnes
# Date: 2009-01-12
#
from ufl import (TestFunction, TrialFunction, VectorElement, dx, grad, inner,
                 tetrahedron, Mesh, FunctionSpace)

element = VectorElement("Lagrange", tetrahedron, 1)
domain = Mesh(VectorElement("Lagrange", tetrahedron, 1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)


def epsilon(v):
    Dv = grad(v)
    return 0.5 * (Dv + Dv.T)


a = inner(epsilon(v), epsilon(u)) * dx
