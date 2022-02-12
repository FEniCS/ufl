#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import *

element = FiniteElement("Lagrange", triangle, 1)

u = TrialFunction(element)
v = TestFunction(element)

a = dot(grad(u), grad(v)) * dx
