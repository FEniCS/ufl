import pytest
from utils import LagrangeElement, MixedElement

import ufl


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, p):
    return epsilon(u) - p * ufl.Identity(u.ufl_shape[0])


@pytest.mark.parametrize("replace_arguments", [True, False])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_extract_blocks(rank, replace_arguments):
    """Test extractions of blocks from mixed function space."""
    cell = ufl.triangle
    domain = ufl.Mesh(LagrangeElement(cell, 1, (2,)))
    fe_scalar = LagrangeElement(cell, 1)
    fe_vector = LagrangeElement(cell, 1, (2,))

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

        # Extracting blocks replaces arguments with `ufl.as_vector`
        if replace_arguments:
            v_ = ufl.TestFunction(V)
            q_ = ufl.TestFunction(Q)
            F_sub = rhs(uh, ph, ufl.as_vector([vi for vi in v_]), q_)
        else:
            F_sub = rhs(uh, ph, ufl.as_vector([vi for vi in v]), q)

        F_0_ext = ufl.extract_blocks(F, 0, replace_argument=replace_arguments)
        F_1_ext = ufl.extract_blocks(F, 1, replace_argument=replace_arguments)
        assert F_sub[0].signature() == F_0_ext.signature()
        assert F_sub[1].signature() == F_1_ext.signature()
    elif rank == 2:

        def lhs(u, p, v, q):
            J_00 = ufl.inner(u, v) * ufl.dx(domain=domain)
            J_01 = ufl.div(v) * p * ufl.dx
            J_10 = q * ufl.div(u) * ufl.dx
            J_11 = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
            return J_00, J_01, J_10, J_11

        v, q = ufl.TestFunctions(W)
        uh, ph = ufl.TrialFunctions(W)
        if replace_arguments:
            u_ = ufl.TrialFunction(V)
            p_ = ufl.TrialFunction(Q)
            v_ = ufl.TestFunction(V)
            q_ = ufl.TestFunction(Q)
            J_sub = lhs(ufl.as_vector([ui for ui in u_]), p_, ufl.as_vector([vi for vi in v_]), q_)
        else:
            J_sub = lhs(ufl.as_vector([ui for ui in uh]), ph, ufl.as_vector([vi for vi in v]), q)

        J = sum(lhs(uh, ph, v, q))

        for i in range(2):
            for j in range(2):
                J_ij_ext = ufl.extract_blocks(J, i, j, replace_argument=replace_arguments)
                assert J_sub[2 * i + j].signature() == J_ij_ext.signature()


def test_postive_restricted_extract_none():
    cell = ufl.triangle
    d = cell.topological_dimension
    domain = ufl.Mesh(LagrangeElement(cell, 1, (d,)))
    el_u = LagrangeElement(cell, 2, (d,))
    el_p = LagrangeElement(cell, 1)
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


def test_part_extract():
    """Test extraction of a single block from a mixed function space form."""
    cell = ufl.quadrilateral
    d = cell.topological_dimension
    domain = ufl.Mesh(LagrangeElement(cell, 1, (d,)))
    el_u = LagrangeElement(cell, 2, (d,))
    el_p = LagrangeElement(cell, 1, (d,))
    S = ufl.FunctionSpace(domain, el_u)
    U = ufl.FunctionSpace(domain, el_p)
    Q = ufl.FunctionSpace(domain, el_u)
    W = ufl.MixedFunctionSpace(S, U, Q)
    wh = ufl.TrialFunctions(W)
    xh = ufl.TestFunctions(W)

    a_ref = [[None for _ in range(3)] for _ in range(3)]
    for i, w_i in enumerate(wh):
        for j, x_j in enumerate(xh):
            a_ji = ufl.inner(w_i, x_j) * ufl.dx
            a_ref[j][i] = a_ji

    # Check that extracting only last element gives an almost empty matrix
    a = a_ref[2][2]
    a_blocked = ufl.extract_blocks(a)
    for i in range(3):
        for j in range(3):
            if (i, j) == (2, 2):
                assert a_blocked[i][j] == a_ref[i][j]
            else:
                assert a_blocked[i][j] is None

    # Extract the the last two rows and the last column and check
    # that we get the expected result
    a2 = sum(a_ref[i][j] for i in [1, 2] for j in [2])
    a2_blocked = ufl.extract_blocks(a2)
    for i in range(3):
        for j in range(3):
            if i < 1 or j < 1 or j < 2:
                assert a2_blocked[i][j] is None
            else:
                assert a2_blocked[i][j] == a_ref[i][j]

    # Extract first element. Check that we only get the (0,0) block
    a1 = a_ref[0][0]
    a1_blocked = ufl.extract_blocks(a1)
    assert len(a1_blocked) == 1
    assert len(a1_blocked[0]) == 1
    assert a1_blocked[0][0] == a_ref[0][0]
