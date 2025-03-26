from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, dx, triangle

element = LagrangeElement(triangle, 1)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)
v = TestFunction(space)
u = TrialFunction(space)
f = Coefficient(space)

a = u * v * dx
L = f * v * dx
