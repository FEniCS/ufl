#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, TestFunction, TrialFunction, derivative, dx, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

v = TestFunction(element)
u = TrialFunction(element)
w = Coefficient(element)

L = w**5 * v * dx
a = derivative(L, w)
