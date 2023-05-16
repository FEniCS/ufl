#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import TestFunction, TrialFunction, dot, dx, grad, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

u = TrialFunction(element)
v = TestFunction(element)

a = dot(grad(u), grad(v)) * dx
