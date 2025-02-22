__authors__ = "Cecile Daversin Catty"
__date__ = "2019-03-26 -- 2019-03-26"

from ufl import (
    Constant,
    FunctionSpace,
    Measure,
    Mesh,
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    dx,
    grad,
    inner,
    interval,
    lhs,
    rhs,
    tetrahedron,
    triangle,
)
from ufl.algorithms import expand_derivatives, renumbering
from ufl.algorithms.formsplitter import extract_blocks
from ufl.finiteelement import FiniteElement
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1


def test_mixed_functionspace(self):
    # Domains
    domain_3d = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1))
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    domain_1d = Mesh(FiniteElement("Lagrange", interval, 1, (1,), identity_pullback, H1))
    # Finite elements
    f_1d = FiniteElement("Lagrange", interval, 1, (), identity_pullback, H1)
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    f_3d = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pullback, H1)
    # Function spaces
    V_3d = FunctionSpace(domain_3d, f_3d)
    V_2d = FunctionSpace(domain_2d, f_2d)
    V_1d = FunctionSpace(domain_1d, f_1d)

    # MixedFunctionSpace = V_3d x V_2d x V_1d
    V = MixedFunctionSpace(V_3d, V_2d, V_1d)
    # Check sub spaces
    assert V.num_sub_spaces() == 3
    assert V.ufl_sub_space(0) == V_3d
    assert V.ufl_sub_space(1) == V_2d
    assert V.ufl_sub_space(2) == V_1d

    # Arguments from MixedFunctionSpace
    (u_3d, u_2d, u_1d) = TrialFunctions(V)
    (v_3d, v_2d, v_1d) = TestFunctions(V)

    # Measures
    dx3 = Measure("dx", domain=domain_3d)
    dx2 = Measure("dx", domain=domain_2d)
    dx1 = Measure("dx", domain=domain_1d)

    # Mixed variational form
    # LHS
    a_11 = u_1d * v_1d * dx1
    a_22 = u_2d * v_2d * dx2
    a_33 = u_3d * v_3d * dx3
    a_21 = u_2d * v_1d * dx1
    a_12 = u_1d * v_2d * dx1
    a_32 = u_3d * v_2d * dx2
    a_23 = u_2d * v_3d * dx2
    a_31 = u_3d * v_1d * dx1
    a_13 = u_1d * v_3d * dx1
    a = a_11 + a_22 + a_33 + a_21 + a_12 + a_32 + a_23 + a_31 + a_13
    # RHS
    f_1 = v_1d * dx1
    f_2 = v_2d * dx2
    f_3 = v_3d * dx3
    f = f_1 + f_2 + f_3

    # Check extract_block algorithm
    # LHS
    assert extract_blocks(a, 0, 0) == a_33
    assert extract_blocks(a, 0, 1) == a_23
    assert extract_blocks(a, 0, 2) == a_13
    assert extract_blocks(a, 1, 0) == a_32
    assert extract_blocks(a, 1, 1) == a_22
    assert extract_blocks(a, 1, 2) == a_12
    assert extract_blocks(a, 2, 0) == a_31
    assert extract_blocks(a, 2, 1) == a_21
    assert extract_blocks(a, 2, 2) == a_11
    # RHS
    assert extract_blocks(f, 0) == f_3
    assert extract_blocks(f, 1) == f_2
    assert extract_blocks(f, 2) == f_1

    # Test dual space method
    V_dual = V.dual()
    assert V_dual.num_sub_spaces() == 3
    assert V_dual.ufl_sub_space(0) == V_3d.dual()
    assert V_dual.ufl_sub_space(1) == V_2d.dual()
    assert V_dual.ufl_sub_space(2) == V_1d.dual()

    V_dual = V.dual(0, 2)
    assert V_dual.num_sub_spaces() == 3
    assert V_dual.ufl_sub_space(0) == V_3d.dual()
    assert V_dual.ufl_sub_space(1) == V_2d
    assert V_dual.ufl_sub_space(2) == V_1d.dual()


def test_lhs_rhs():
    V = FiniteElement("Lagrange", interval, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", interval, 1, (1,), identity_pullback, H1))
    space0 = FunctionSpace(domain, V)
    space1 = FunctionSpace(domain, V)
    mixed_space = MixedFunctionSpace(space0, space1)

    def mass(u, v):
        return inner(u, v) * dx

    def mixed(u, v):
        return inner(u.dx(0), v) * dx

    def stiffness(u, v):
        return inner(grad(u), grad(v)) * dx

    def source1(f, v):
        return f * v * dx

    def source2(f, v):
        return f * v.dx(0) * dx

    u, p = TrialFunctions(mixed_space)
    v, q = TestFunctions(mixed_space)
    f = Constant(domain)
    g = Constant(domain)
    F = mass(u, v) + mixed(u, q) + stiffness(p, q) + source1(f, v) + source2(g, q)

    a = lhs(F)
    a_blocked = extract_blocks(a)
    L = rhs(F)
    L_blocked = extract_blocks(L)

    assert a_blocked[0][0] == mass(u, v)
    assert a_blocked[0][1] is None
    assert a_blocked[1][0] == mixed(u, q)
    assert renumbering.renumber_indices(a_blocked[1][1]) == renumbering.renumber_indices(
        expand_derivatives(stiffness(p, q))
    )

    assert L_blocked[0] == -source1(f, v)
    assert L_blocked[1] == -source2(g, q)
