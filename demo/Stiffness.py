#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction, VectorElement, dot, dx, grad, triangle

element = FiniteElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)

u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
