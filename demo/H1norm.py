#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FiniteElement, dot, dx, grad, triangle, Mesh, FunctionSpace, VectorElement

element = FiniteElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)

f = Coefficient(space)

a = (f * f + dot(grad(f), grad(f))) * dx
