import pytest

import ufl
import ufl.algorithms
from ufl.finiteelement import FiniteElement, MixedElement


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, p):
    return epsilon(u) - p * ufl.Identity(u.ufl_shape[0])


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_extract_blocks(rank):
    """Test extractions of blocks from mixed function space."""
    cell = ufl.triangle
    domain = ufl.Mesh(FiniteElement("Lagrange", cell, 1, (2,), ufl.identity_pullback, ufl.H1))
    fe_scalar = FiniteElement("Lagrange", cell, 1, (), ufl.identity_pullback, ufl.H1)
    fe_vector = FiniteElement("Lagrange", cell, 1, (2,), ufl.identity_pullback, ufl.H1)

    me = MixedElement([fe_vector, fe_scalar])

    # # Function spaces
    W = ufl.FunctionSpace(domain, me)
    V = ufl.FunctionSpace(domain, fe_vector)
    Q = ufl.FunctionSpace(domain, fe_scalar)

    if rank == 0:
        wh = ufl.Coefficient(W)
        uh, ph = ufl.split(wh)
        # Test that functionals return the identity
        J = ufl.inner(sigma(uh, ph), sigma(uh, ph)) * ufl.dx
        J0 = ufl.extract_blocks(J, 0)
        assert len(J0) == 1
        assert J == J0[0]
    elif rank == 1:

        def rhs(uh, ph, v, q):
            F_0 = ufl.inner(sigma(uh, ph), epsilon(v)) * ufl.dx(domain=domain)
            F_1 = ufl.div(uh) * q * ufl.dx
            return F_0, F_1

        wh = ufl.Coefficient(W)
        uh, ph = ufl.split(wh)
        v, q = ufl.TestFunctions(W)
        F = sum(rhs(uh, ph, v, q))

        v_ = ufl.TestFunction(V)
        q_ = ufl.TestFunction(Q)
        F_sub = rhs(uh, ph, ufl.as_vector([vi for vi in v_]), q_)

        F_0_ext = ufl.extract_blocks(F, 0)
        assert F_sub[0].signature() == F_0_ext.signature()

        F_1_ext = ufl.extract_blocks(F, 1)
        assert F_sub[1].signature() == F_1_ext.signature()
    elif rank == 2:

        def lhs(u, p, v, q):
            J_00 = ufl.inner(u, v) * ufl.dx(domain=domain)
            J_01 = ufl.div(v) * p * ufl.dx
            J_10 = q * ufl.div(u) * ufl.dx
            J_11 = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
            return J_00, J_01, J_10, J_11

        v_ = ufl.TestFunction(V)
        q_ = ufl.TestFunction(Q)
        u_ = ufl.TrialFunction(V)
        p_ = ufl.TrialFunction(Q)
        J_sub = lhs(ufl.as_vector([ui for ui in u_]), p_, ufl.as_vector([vi for vi in v_]), q_)

        v, q = ufl.TestFunctions(W)
        uh, ph = ufl.TrialFunctions(W)
        J = sum(lhs(uh, ph, v, q))

        for i in range(2):
            for j in range(2):
                J_ij_ext = ufl.extract_blocks(J, i, j)
                assert J_sub[2 * i + j].signature() == J_ij_ext.signature()


def test_postive_restricted_extract_none():
    cell = ufl.triangle
    d = cell.topological_dimension()
    domain = ufl.Mesh(FiniteElement("Lagrange", cell, 1, (d,), ufl.identity_pullback, ufl.H1))
    el_u = FiniteElement("Lagrange", cell, 2, (d,), ufl.identity_pullback, ufl.H1)
    el_p = FiniteElement("Lagrange", cell, 1, (), ufl.identity_pullback, ufl.H1)
    V = ufl.FunctionSpace(domain, el_u)
    Q = ufl.FunctionSpace(domain, el_p)
    W = ufl.MixedFunctionSpace(V, Q)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    a = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        + ufl.div(v) * p * ufl.dx
    )
    a += ufl.inner(u("+"), v("+")) * ufl.dS
    a_blocks = ufl.extract_blocks(a)
    assert a_blocks[1][1] is None
