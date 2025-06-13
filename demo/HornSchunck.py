#
# Implemented by imitation of
#  http://code.google.com/p/debiosee/wiki/DemosOptiocFlowHornSchunck
# but not tested so this could contain errors!
#
from utils import LagrangeElement

from ufl import (
    Coefficient,
    Constant,
    FunctionSpace,
    Mesh,
    derivative,
    dot,
    dx,
    grad,
    inner,
    triangle,
)

# Finite element spaces for scalar and vector fields
cell = triangle
S = LagrangeElement(cell, 1)
V = LagrangeElement(cell, 1, (2,))
domain = Mesh(LagrangeElement(cell, 1, (2,)))
S_space = FunctionSpace(domain, S)
V_space = FunctionSpace(domain, V)

# Optical flow function
u = Coefficient(V_space)

# Previous image brightness
I0 = Coefficient(S_space)

# Current image brightness
I1 = Coefficient(S_space)

# Regularization parameter
lamda = Constant(domain)

# Coefficiental to minimize
M = (dot(u, grad(I1)) + (I1 - I0)) ** 2 * dx + lamda * inner(grad(u), grad(u)) * dx

# Derived linear system
L = derivative(M, u)
a = derivative(L, u)
L = -L
