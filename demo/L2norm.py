#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FunctionSpace, Mesh, dx, triangle
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
space = FunctionSpace(domain, element)

f = Coefficient(space)

a = f**2 * dx
