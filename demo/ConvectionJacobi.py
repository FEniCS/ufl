#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, FunctionSpace, Mesh, TestFunction, TrialFunction, dot, dx, grad, triangle
from ufl.finiteelement import FiniteElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), identity_pull_back, H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), identity_pull_back, H1))
space = FunctionSpace(domain, element)

u = TrialFunction(space)
v = TestFunction(space)
w = Coefficient(space)

a = dot(dot(u, grad(w)) + dot(w, grad(u)), v) * dx
