#
# Harmonic map demo using one mixed function u to represent x and y.
# Author: Martin Alnes
# Date: 2009-04-09
#
from ufl import (Coefficient, FiniteElement, VectorElement, derivative, dot,
                 dx, grad, inner, split, triangle, Mesh, FunctionSpace)

cell = triangle
X = VectorElement("Lagrange", cell, 1)
Y = FiniteElement("Lagrange", cell, 1)
M = X * Y
domain = Mesh(VectorElement("Lagrange", cell, 1))
space = FunctionSpace(domain, M)

u = Coefficient(space)
x, y = split(u)

L = inner(grad(x), grad(x)) * dx + dot(x, x) * y * dx

F = derivative(L, u)
J = derivative(F, u)

forms = [L, F, J]
