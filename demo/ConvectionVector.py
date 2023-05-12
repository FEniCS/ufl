#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl_legacy import (Coefficient, TestFunction, VectorElement, dot, dx, grad,
                        triangle)

element = VectorElement("Lagrange", triangle, 1)

v = TestFunction(element)
w = Coefficient(element)

a = dot(dot(w, grad(w)), v) * dx
