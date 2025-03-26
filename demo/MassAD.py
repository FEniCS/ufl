#
# Author: Martin Sandve Alnes
# Date: 2008-10-28
#
from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, derivative, dx, triangle

element = LagrangeElement(triangle, 1)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

u = Coefficient(space)

# L2 norm
M = u**2 / 2 * dx
# source vector
L = derivative(M, u)
# mass matrix
a = derivative(L, u)
