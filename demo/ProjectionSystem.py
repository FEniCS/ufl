from ufl import Coefficient, TestFunction, TrialFunction, dx, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
v = TestFunction(element)
u = TrialFunction(element)
f = Coefficient(element)

a = u * v * dx
L = f * v * dx
