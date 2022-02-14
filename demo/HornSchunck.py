#
# Implemented by imitation of
#  http://code.google.com/p/debiosee/wiki/DemosOptiocFlowHornSchunck
# but not tested so this could contain errors!
#
from ufl import (Coefficient, Constant, FiniteElement, VectorElement,
                 derivative, dot, dx, grad, inner, triangle)

# Finite element spaces for scalar and vector fields
cell = triangle
S = FiniteElement("CG", cell, 1)
V = VectorElement("CG", cell, 1)

# Optical flow function
u = Coefficient(V)

# Previous image brightness
I0 = Coefficient(S)

# Current image brightness
I1 = Coefficient(S)

# Regularization parameter
lamda = Constant(cell)

# Coefficiental to minimize
M = (dot(u, grad(I1)) + (I1 - I0))**2 * dx\
    + lamda * inner(grad(u), grad(u)) * dx

# Derived linear system
L = derivative(M, u)
a = derivative(L, u)
L = -L
