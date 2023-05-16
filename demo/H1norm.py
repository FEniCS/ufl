#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, dot, dx, grad, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

f = Coefficient(element)

a = (f * f + dot(grad(f), grad(f))) * dx
