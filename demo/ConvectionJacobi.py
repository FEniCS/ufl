#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import (Coefficient, TestFunction, TrialFunction, dot, dx, grad,
                 triangle)
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)

u = TrialFunction(element)
v = TestFunction(element)
w = Coefficient(element)

a = dot(dot(u, grad(w)) + dot(w, grad(u)), v) * dx
