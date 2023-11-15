#
# Author: Martin Sandve Alnes
# Date: 2008-10-28
#
from ufl import Coefficient, FunctionSpace, Mesh, derivative, dx, triangle
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1

element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pullback, H1))
space = FunctionSpace(domain, element)

u = Coefficient(space)

# L2 norm
M = u**2 / 2 * dx
# source vector
L = derivative(M, u)
# mass matrix
a = derivative(L, u)
