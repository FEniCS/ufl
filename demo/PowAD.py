#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import (Coefficient, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction, VectorElement,
                 derivative, dx, triangle)

element = FiniteElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
w = Coefficient(space)

L = w**5 * v * dx
a = derivative(L, w)
