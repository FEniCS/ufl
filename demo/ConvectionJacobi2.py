#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, VectorElement, dx, i, j, triangle

element = VectorElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)

u = TrialFunction(space)
v = TestFunction(space)
w = Coefficient(space)

a = (u[j] * w[i].dx(j) + w[j] * u[i].dx(j)) * v[i] * dx
