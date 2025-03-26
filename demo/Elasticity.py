#
# Author: Anders Logg
# Modified by: Martin Sandve Alnes
# Date: 2009-01-12
#
from utils import LagrangeElement

from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, dx, grad, inner, tetrahedron

element = LagrangeElement(tetrahedron, 1, (3,))
domain = Mesh(LagrangeElement(tetrahedron, 1, (3,)))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)


def epsilon(v):
    Dv = grad(v)
    return 0.5 * (Dv + Dv.T)


a = inner(epsilon(v), epsilon(u)) * dx
