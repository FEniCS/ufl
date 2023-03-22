#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl_legacy import (Coefficient, TestFunction, TrialFunction, VectorElement, dx,
                        i, j, triangle)

element = VectorElement("Lagrange", triangle, 1)

u = TrialFunction(element)
v = TestFunction(element)
w = Coefficient(element)

a = (u[j] * w[i].dx(j) + w[j] * u[i].dx(j)) * v[i] * dx
