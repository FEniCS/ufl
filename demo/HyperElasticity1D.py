#
# Author: Martin Sandve Alnes
# Date: 2008-10-03
#
from ufl import (Coefficient, Constant, derivative, dx, exp,
                 interval, variable)
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1

cell = interval
element = FiniteElement("Lagrange", cell, 2, (), (), "identity", H1)
u = Coefficient(element)
b = Constant(cell)
K = Constant(cell)

E = u.dx(0) + u.dx(0)**2 / 2
E = variable(E)
Q = b * E**2
psi = K * (exp(Q) - 1)

f = psi * dx
F = derivative(f, u)
J = derivative(-F, u)

forms = [f, F, J]
