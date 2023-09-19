#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FiniteElement, dx, triangle, Mesh, FunctionSpace, VectorElement

element = FiniteElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)

f = Coefficient(space)

a = f**2 * dx
