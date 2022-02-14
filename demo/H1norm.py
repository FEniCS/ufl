#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FiniteElement, dot, dx, grad, triangle

element = FiniteElement("Lagrange", triangle, 1)

f = Coefficient(element)

a = (f * f + dot(grad(f), grad(f))) * dx
