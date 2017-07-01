#
# Author: Martin Sandve Alnes
# Date: 2008-12-22
#

# Modified by Garth N. Wells, 2009

import ufl

def build_forms():
    # Cell and its properties
    cell = ufl.tetrahedron
    d = cell.geometric_dimension()
    N = ufl.FacetNormal(cell)
    x = ufl.SpatialCoordinate(cell)

    # Elements
    u_element = ufl.VectorElement("CG", cell, 2)
    p_element = ufl.FiniteElement("CG", cell, 1)
    A_element = ufl.TensorElement("CG", cell, 1)

    # Test and trial functions
    v = ufl.TestFunction(u_element)
    w = ufl.TrialFunction(u_element)

    # Displacement at current and two previous timesteps
    u   = ufl.Coefficient(u_element)
    up  = ufl.Coefficient(u_element)
    upp = ufl.Coefficient(u_element)

    # Time parameters
    dt = ufl.Constant(cell)

    # Fiber field
    A = ufl.Coefficient(A_element)

    # External forces
    T = ufl.Coefficient(u_element)
    p0 = ufl.Coefficient(p_element)

    # Material parameters FIXME
    rho = ufl.Constant(cell)
    K   = ufl.Constant(cell)
    c00 = ufl.Constant(cell)
    c11 = ufl.Constant(cell)
    c22 = ufl.Constant(cell)

    # Deformation gradient
    I = ufl.Identity(d)
    F = I + ufl.grad(u)
    F = ufl.variable(F)
    Finv = ufl.inv(F)
    J = ufl.det(F)

    # Left Cauchy-Green deformation tensor
    B = F*F.T
    I1_B = ufl.tr(B)
    I2_B = (I1_B**2 - ufl.tr(B*B))/2
    I3_B = J**2

    # Right Cauchy-Green deformation tensor
    C = F.T*F
    I1_C = ufl.tr(C)
    I2_C = (I1_C**2 - ufl.tr(C*C))/2
    I3_C = J**2

    # Green strain tensor
    E = (C-I)/2

    # Mapping of strain in fiber directions
    Ef = A*E*A.T

    # Strain energy function W(Q(Ef))
    Q = c00*Ef[0,0]**2 + c11*Ef[1,1]**2 + c22*Ef[2,2]**2 # FIXME: insert some simple law here
    W = (K/2)*(ufl.exp(Q) - 1) # + p stuff

    # First Piola-Kirchoff stress tensor
    P = ufl.diff(W, F)

    # Acceleration term discretized with finite differences
    k = dt/rho
    acc = (u - 2*up + upp)

    # Residual equation # FIXME: Can contain errors, not tested!
    a_F =   ufl.inner(acc, v)*ufl.dx \
            + k*ufl.inner(P, ufl.grad(v))*ufl.dx \
            - k*ufl.dot(J*Finv*T, v)*ufl.ds(0) \
            - k*ufl.dot(J*Finv*p0*N, v)*ufl.ds(1)

    # Jacobi matrix of residual equation
    a_J = ufl.derivative(a_F, u, w)

    # Export forms
    forms = [a_F, a_J]

forms = [build_forms() for i in range(1000)]
