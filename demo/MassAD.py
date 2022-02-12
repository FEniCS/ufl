#
# Author: Martin Sandve Alnes
# Date: 2008-10-28
#
from ufl import *

element = FiniteElement("Lagrange", triangle, 1)

u = Coefficient(element)

# L2 norm
M = u**2 / 2 * dx
# source vector
L = derivative(M, u)
# mass matrix
a = derivative(L, u)
