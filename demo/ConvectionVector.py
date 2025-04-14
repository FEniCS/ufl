#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, dot, dx, grad, triangle

element = LagrangeElement(triangle, 1, (2,))
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

v = TestFunction(space)
w = Coefficient(space)

a = dot(dot(w, grad(w)), v) * dx
