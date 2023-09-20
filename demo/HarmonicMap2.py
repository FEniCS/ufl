#
# Harmonic map demo using one mixed function u to represent x and y.
# Author: Martin Alnes
# Date: 2009-04-09
#
from ufl import Coefficient, FunctionSpace, Mesh, derivative, dot, dx, grad, inner, split, triangle
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.sobolevspace import H1

cell = triangle
X = FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1)
Y = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
M = MixedElement([X, Y])
domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
space = FunctionSpace(domain, M)

u = Coefficient(space)
x, y = split(u)

L = inner(grad(x), grad(x)) * dx + dot(x, x) * y * dx

F = derivative(L, u)
J = derivative(F, u)

forms = [L, F, J]
