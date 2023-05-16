#
# Harmonic map demo using separate coefficients x and y.
# Author: Martin Alnes
# Date: 2009-04-09
#
from ufl import Coefficient, derivative, dot, dx, grad, inner, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

cell = triangle
X = FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1)
Y = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)

x = Coefficient(X)
y = Coefficient(Y)

L = inner(grad(x), grad(x)) * dx + dot(x, x) * y * dx

F = derivative(L, (x, y))
J = derivative(F, (x, y))

forms = [L, F, J]
