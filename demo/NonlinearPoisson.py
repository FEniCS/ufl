from utils import LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    dot,
    dx,
    grad,
    triangle,
)

element = LagrangeElement(triangle, 1)
domain = Mesh(LagrangeElement(triangle, 1, (2,)))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
u0 = Coefficient(space)
f = Coefficient(space)

a = (1 + u0**2) * dot(grad(v), grad(u)) * dx + 2 * u0 * u * dot(grad(v), grad(u0)) * dx
L = v * f * dx - (1 + u0**2) * dot(grad(v), grad(u0)) * dx
