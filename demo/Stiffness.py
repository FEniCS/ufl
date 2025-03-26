#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from utils import LagrangeElement

from ufl import FunctionSpace, Mesh, TestFunction, TrialFunction, dot, dx, grad, triangle

element = LagrangeElement(triangle, 1)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
