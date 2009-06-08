

from ufl import *

e1 = FiniteElement("CG", triangle, 1)
e2 = FiniteElement("CG", triangle, 2)
m = e1 + e2

# Dummy functions
g = Function(e1)
f = Function(e1)

# Mixed function, main unknown of nonlinear system
kc = Function(m)
k, c = split(kc)

# Functional
M = k**2*c**2*dx

# Correct, makes basis function automatically:
L1 = derivative(M, kc)

# Correct, providing basis function manually:
dkc = BasisFunction(m)
dk, dc = split(dkc)
L2 = derivative(M, kc, dkc) + f*dk*dx + g*dc*dx

# Doesn't work, expecting BasisFunction, not Indexed:
L3 = derivative(M, k, dk) + derivative(M, c, dc)

