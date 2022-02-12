#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import *

element = VectorElement("Lagrange", triangle, 1)

u = TrialFunction(element)
v = TestFunction(element)
w = Coefficient(element)

a = dot(dot(u, grad(w)) + dot(w, grad(u)), v) * dx
