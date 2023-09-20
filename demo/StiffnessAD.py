#
# Author: Martin Sandve Alnes
# Date: 2008-10-30
#
from ufl import Coefficient, FiniteElement, FunctionSpace, Mesh, action, adjoint, derivative, dx, grad, inner, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
space = FunctionSpace(domain, element)

w = Coefficient(space)

# H1 semi-norm
f = inner(grad(w), grad(w)) / 2 * dx
# grad(w) : grad(v)
b = derivative(f, w)
# stiffness matrix, grad(u) : grad(v)
a = derivative(b, w)

# adjoint, grad(v) : grad(u)
astar = adjoint(a)
# action of adjoint, grad(v) : grad(w)
astaraction = action(astar)

forms = [f, b, a, astar, astaraction]
