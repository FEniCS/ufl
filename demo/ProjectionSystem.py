from ufl import (Coefficient, FiniteElement, TestFunction, TrialFunction, dx,
                 triangle)

element = FiniteElement("Lagrange", triangle, 1)
v = TestFunction(element)
u = TrialFunction(element)
f = Coefficient(element)

a = u * v * dx
L = f * v * dx
