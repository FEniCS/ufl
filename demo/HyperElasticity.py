#
# Author: Martin Sandve Alnes
# Date: 2008-12-22
#

from utils import LagrangeElement

from ufl import (
    Coefficient,
    Constant,
    FacetNormal,
    FunctionSpace,
    Identity,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    derivative,
    det,
    diff,
    dot,
    ds,
    dx,
    exp,
    grad,
    inner,
    inv,
    tetrahedron,
    tr,
    variable,
)

# Modified by Garth N. Wells, 2009

# Cell and its properties
cell = tetrahedron
domain = Mesh(LagrangeElement(cell, 1, (3,)))
d = 3
N = FacetNormal(domain)
x = SpatialCoordinate(domain)

# Elements
u_element = LagrangeElement(cell, 2, (3,))
p_element = LagrangeElement(cell, 1)
A_element = LagrangeElement(cell, 1, (3, 3))

# Spaces
u_space = FunctionSpace(domain, u_element)
p_space = FunctionSpace(domain, p_element)
A_space = FunctionSpace(domain, A_element)

# Test and trial functions
v = TestFunction(u_space)
w = TrialFunction(u_space)

# Displacement at current and two previous timesteps
u = Coefficient(u_space)
up = Coefficient(u_space)
upp = Coefficient(u_space)

# Time parameters
dt = Constant(domain)

# Fiber field
A = Coefficient(A_space)

# External forces
T = Coefficient(u_space)
p0 = Coefficient(p_space)

# Material parameters FIXME
rho = Constant(domain)
K = Constant(domain)
c00 = Constant(domain)
c11 = Constant(domain)
c22 = Constant(domain)

# Deformation gradient
Id = Identity(d)
F = Id + grad(u)
F = variable(F)
Finv = inv(F)
J = det(F)

# Left Cauchy-Green deformation tensor
B = F * F.T
I1_B = tr(B)
I2_B = (I1_B**2 - tr(B * B)) / 2
I3_B = J**2

# Right Cauchy-Green deformation tensor
C = F.T * F
I1_C = tr(C)
I2_C = (I1_C**2 - tr(C * C)) / 2
I3_C = J**2

# Green strain tensor
E = (C - Id) / 2

# Mapping of strain in fiber directions
Ef = A * E * A.T

# Strain energy function W(Q(Ef))
Q = (
    c00 * Ef[0, 0] ** 2 + c11 * Ef[1, 1] ** 2 + c22 * Ef[2, 2] ** 2
)  # FIXME: insert some simple law here
W = (K / 2) * (exp(Q) - 1)  # + p stuff

# First Piola-Kirchoff stress tensor
P = diff(W, F)

# Acceleration term discretized with finite differences
k = dt / rho
acc = u - 2 * up + upp

# Residual equation # FIXME: Can contain errors, not tested!
a_F = (
    inner(acc, v) * dx
    + k * inner(P, grad(v)) * dx
    - k * dot(J * Finv * T, v) * ds(0)
    - k * dot(J * Finv * p0 * N, v) * ds(1)
)

# Jacobi matrix of residual equation
a_J = derivative(a_F, u, w)

# Export forms
forms = [a_F, a_J]
