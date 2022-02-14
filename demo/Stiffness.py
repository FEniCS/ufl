#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import (FiniteElement, TestFunction, TrialFunction, dot, dx, grad,
                 triangle)

element = FiniteElement("Lagrange", triangle, 1)

u = TrialFunction(element)
v = TestFunction(element)

a = dot(grad(u), grad(v)) * dx
