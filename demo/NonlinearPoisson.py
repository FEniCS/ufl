from ufl import (Coefficient, TestFunction, TrialFunction, dot, dx, grad,
                 triangle)
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

v = TestFunction(element)
u = TrialFunction(element)
u0 = Coefficient(element)
f = Coefficient(element)

a = (1 + u0**2) * dot(grad(v), grad(u)) * dx \
    + 2 * u0 * u * dot(grad(v), grad(u0)) * dx
L = v * f * dx - (1 + u0**2) * dot(grad(v), grad(u0)) * dx
