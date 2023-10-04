from ufl import (Coefficient, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction, VectorElement, dx,
                 triangle)

element = FiniteElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)
v = TestFunction(space)
u = TrialFunction(space)
f = Coefficient(space)

a = u * v * dx
L = f * v * dx
