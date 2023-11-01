#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, derivative, dx, triangle
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
space = FunctionSpace(domain, element)

v = TestFunction(space)
u = TrialFunction(space)
w = Coefficient(space)

L = w**5 * v * dx
a = derivative(L, w)
