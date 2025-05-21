#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from utils import LagrangeElement

from ufl import Coefficient, Constant, FunctionSpace, Mesh, derivative, dx, exp, interval, variable

cell = interval
element = LagrangeElement(cell, 2)
domain = Mesh(LagrangeElement(cell, 1, (1,)))
space = FunctionSpace(domain, element)

u = Coefficient(space)
b = Constant(domain)
K = Constant(domain)

E = u.dx(0) + u.dx(0) ** 2 / 2
E = variable(E)
Q = b * E**2
psi = K * (exp(Q) - 1)

f = psi * dx
F = derivative(f, u)
J = derivative(-F, u)

forms = [f, F, J]
