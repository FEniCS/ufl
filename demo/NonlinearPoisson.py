from ufl import (Coefficient, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction, VectorElement, dot, dx,
                 grad, triangle)

element = FiniteElement("Lagrange", triangle, 1)
domain = Mesh(VectorElement("Lagrange", triangle, 1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
u0 = Coefficient(space)
f = Coefficient(space)

a = (1 + u0**2) * dot(grad(v), grad(u)) * dx \
    + 2 * u0 * u * dot(grad(v), grad(u0)) * dx
L = v * f * dx - (1 + u0**2) * dot(grad(v), grad(u0)) * dx
