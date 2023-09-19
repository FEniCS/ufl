#
# Author: Martin Sandve Alnes
# Date: 2008-10-28
#
from ufl import Coefficient, FiniteElement, derivative, dx, triangle, Mesh, FunctionSpace, VectorElement

element = FiniteElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)

u = Coefficient(space)

# L2 norm
M = u**2 / 2 * dx
# source vector
L = derivative(M, u)
# mass matrix
a = derivative(L, u)
