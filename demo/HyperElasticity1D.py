#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import Coefficient, Constant, FunctionSpace, Mesh, derivative, dx, exp, interval, variable
from ufl.finiteelement import FiniteElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1

cell = interval
element = FiniteElement("Lagrange", cell, 2, (), identity_pull_back, H1)
domain = Mesh(FiniteElement("Lagrange", cell, 1, (1, ), identity_pull_back, H1))
space = FunctionSpace(domain, element)

u = Coefficient(space)
b = Constant(domain)
K = Constant(domain)

E = u.dx(0) + u.dx(0)**2 / 2
E = variable(E)
Q = b * E**2
psi = K * (exp(Q) - 1)

f = psi * dx
F = derivative(f, u)
J = derivative(-F, u)

forms = [f, F, J]
