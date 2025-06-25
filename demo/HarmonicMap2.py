#
# Harmonic map demo using one mixed function u to represent x and y.
# Author: Martin Alnes
# Date: 2009-04-09
#
from utils import LagrangeElement, MixedElement

from ufl import Coefficient, FunctionSpace, Mesh, derivative, dot, dx, grad, inner, split, triangle

cell = triangle
X = LagrangeElement(cell, 1, (2,))
Y = LagrangeElement(cell, 1)
M = MixedElement([X, Y])
domain = Mesh(LagrangeElement(cell, 1, (2,)))
space = FunctionSpace(domain, M)

u = Coefficient(space)
x, y = split(u)

L = inner(grad(x), grad(x)) * dx + dot(x, x) * y * dx

F = derivative(L, u)
J = derivative(F, u)

forms = [L, F, J]
