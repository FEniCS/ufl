#
# Harmonic map demo using one mixed function u to represent x and y.
# Author: Martin Alnes
# Date: 2009-04-09
#
from ufl import (Coefficient, FiniteElement, VectorElement, derivative, dot,
                 dx, grad, inner, split, triangle)

cell = triangle
X = VectorElement("Lagrange", cell, 1)
Y = FiniteElement("Lagrange", cell, 1)
M = X * Y

u = Coefficient(M)
x, y = split(u)

L = inner(grad(x), grad(x)) * dx + dot(x, x) * y * dx

F = derivative(L, u)
J = derivative(F, u)

forms = [L, F, J]
