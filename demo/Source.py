#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import *

element = FiniteElement("Lagrange", triangle, 1)

v = TestFunction(element)
f = Coefficient(element)

a = f * v * dx
