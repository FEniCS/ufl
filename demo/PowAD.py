#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl_legacy import (Coefficient, FiniteElement, TestFunction, TrialFunction,
                        derivative, dx, triangle)

element = FiniteElement("Lagrange", triangle, 1)

v = TestFunction(element)
u = TrialFunction(element)
w = Coefficient(element)

L = w**5 * v * dx
a = derivative(L, w)
