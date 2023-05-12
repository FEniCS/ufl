#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl_legacy import (Coefficient, TestFunction, TrialFunction, VectorElement, dot,
                        dx, grad, triangle)

element = VectorElement("Lagrange", triangle, 1)

u = TrialFunction(element)
v = TestFunction(element)
w = Coefficient(element)

a = dot(dot(u, grad(w)) + dot(w, grad(u)), v) * dx
