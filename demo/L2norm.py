#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FiniteElement, dx, triangle

element = FiniteElement("Lagrange", triangle, 1)

f = Coefficient(element)

a = f**2 * dx
