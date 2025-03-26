#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, dx, i, j, triangle

element = LagrangeElement(triangle, 1, (2,))
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

u = TrialFunction(space)
v = TestFunction(space)
w = Coefficient(space)

a = (u[j] * w[i].dx(j) + w[j] * u[i].dx(j)) * v[i] * dx
