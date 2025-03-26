#
# Harmonic map demo using separate coefficients x and y.
# Author: Martin Alnes
# Date: 2009-04-09
#
from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, derivative, dot, dx, grad, inner, triangle

cell = triangle
X = LagrangeElement(cell, 1, (2,))
Y = LagrangeElement(cell, 1)
domain = Mesh(LagrangeElement(cell, 1, (2,)))
X_space = FunctionSpace(domain, X)
Y_space = FunctionSpace(domain, Y)

x = Coefficient(X_space)
y = Coefficient(Y_space)

L = inner(grad(x), grad(x)) * dx + dot(x, x) * y * dx

F = derivative(L, (x, y))
J = derivative(F, (x, y))

forms = [L, F, J]
