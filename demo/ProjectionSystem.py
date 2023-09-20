from ufl import Coefficient, FiniteElement, FunctionSpace, Mesh, TestFunction, TrialFunction, dx, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
space = FunctionSpace(domain, element)
v = TestFunction(space)
u = TrialFunction(space)
f = Coefficient(space)

a = u * v * dx
L = f * v * dx
