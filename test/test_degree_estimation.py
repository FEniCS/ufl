__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-03-12 -- 2009-01-28"


from ufl import (Argument, Coefficient, Coefficients, FacetNormal, FiniteElement, FunctionSpace, Mesh,
                 SpatialCoordinate, TensorProductElement, VectorElement, cos, div, dot, grad, i, inner, nabla_div,
                 nabla_grad, sin, tan, triangle)
from ufl.algorithms import estimate_total_polynomial_degree


def test_total_degree_estimation():
    V1 = FiniteElement("CG", triangle, 1)
    V2 = FiniteElement("CG", triangle, 2)
    VV = VectorElement("CG", triangle, 3)
    VM = V1 * V2
    O1 = TensorProductElement(V1, V1)
    O2 = TensorProductElement(V2, V1)

    domain = Mesh(VectorElement("Lagrange", triangle, 1))
    tensor_domain = Mesh(VectorElement("Lagrange", O1.cell(), 1))

    v1_space = FunctionSpace(domain, V1)
    v2_space = FunctionSpace(domain, V2)
    vv_space = FunctionSpace(domain, VV)
    vm_space = FunctionSpace(domain, VM)
    o1_space = FunctionSpace(tensor_domain, O1)
    o2_space = FunctionSpace(tensor_domain, O2)

    v1 = Argument(v1_space, 2)
    v2 = Argument(v2_space, 3)
    f1, f2 = Coefficients(vm_space)
    u1 = Coefficient(o1_space)
    u2 = Coefficient(o2_space)
    vv = Argument(vv_space, 4)
    vu = Argument(vv_space, 5)

    x, y = SpatialCoordinate(domain)
    assert estimate_total_polynomial_degree(x) == 1
    assert estimate_total_polynomial_degree(x * y) == 2
    assert estimate_total_polynomial_degree(x ** 3) == 3
    assert estimate_total_polynomial_degree(x ** 3) == 3
    assert estimate_total_polynomial_degree((x - 1) ** 4) == 4

    assert estimate_total_polynomial_degree(vv[0]) == 3
    assert estimate_total_polynomial_degree(v2 * vv[0]) == 5
    assert estimate_total_polynomial_degree(vu[0] * vv[0]) == 6
    assert estimate_total_polynomial_degree(vu[i] * vv[i]) == 6

    assert estimate_total_polynomial_degree(v1) == 1
    assert estimate_total_polynomial_degree(v2) == 2

    # TODO: This should be 1, but 2 is expected behaviour now
    # because f1 is part of a mixed element with max degree 2.
    assert estimate_total_polynomial_degree(f1) == 2

    assert estimate_total_polynomial_degree(f2) == 2
    assert estimate_total_polynomial_degree(v2 * v1) == 3

    # TODO: This should be 2, but 3 is expected behaviour now
    # because f1 is part of a mixed element with max degree 2.
    assert estimate_total_polynomial_degree(f1 * v1) == 3

    assert estimate_total_polynomial_degree(f2 * v1) == 3
    assert estimate_total_polynomial_degree(f2 * v2 * v1) == 5

    assert estimate_total_polynomial_degree(f2 + 3) == 2
    assert estimate_total_polynomial_degree(f2 * 3) == 2
    assert estimate_total_polynomial_degree(f2 ** 3) == 6
    assert estimate_total_polynomial_degree(f2 / 3) == 2
    assert estimate_total_polynomial_degree(f2 / v2) == 4
    assert estimate_total_polynomial_degree(f2 / (x - 1)) == 3

    assert estimate_total_polynomial_degree(v1.dx(0)) == 0
    assert estimate_total_polynomial_degree(f2.dx(0)) == 1

    assert estimate_total_polynomial_degree(f2 * v2.dx(0) * v1.dx(0)) == 2 + 1

    assert estimate_total_polynomial_degree(f2) == 2
    assert estimate_total_polynomial_degree(f2 ** 2) == 4
    assert estimate_total_polynomial_degree(f2 ** 3) == 6
    assert estimate_total_polynomial_degree(f2 ** 3 * v1) == 7
    assert estimate_total_polynomial_degree(f2 ** 3 * v1 + f1 * v1) == 7

    # outer product tuple-degree tests
    assert estimate_total_polynomial_degree(u1) == (1, 1)
    assert estimate_total_polynomial_degree(u2) == (2, 1)
    # derivatives should do nothing (don't know in which direction they act)
    assert estimate_total_polynomial_degree(grad(u2)) == (2, 1)
    assert estimate_total_polynomial_degree(u1 * u1) == (2, 2)
    assert estimate_total_polynomial_degree(u2 * u1) == (3, 2)
    assert estimate_total_polynomial_degree(u2 * u2) == (4, 2)
    assert estimate_total_polynomial_degree(u1 ** 3) == (3, 3)
    assert estimate_total_polynomial_degree(u1 ** 3 + u2 * u2) == (4, 3)
    assert estimate_total_polynomial_degree(u2 ** 2 * u1) == (5, 3)

    # Math functions of constant values are constant values
    nx, ny = FacetNormal(domain)
    e = nx ** 2
    for f in [sin, cos, tan, abs, lambda z:z**7]:
        assert estimate_total_polynomial_degree(f(e)) == 0

    # Based on the arbitrary chosen math function heuristics...
    heuristic_add = 2
    e = x**3
    for f in [sin, cos, tan]:
        assert estimate_total_polynomial_degree(f(e)) == 3 + heuristic_add


def test_some_compound_types():

    # NB! Although some compound types are supported here,
    # some derivatives and compounds must be preprocessed
    # prior to degree estimation. In generic code, this algorithm
    # should only be applied after preprocessing.

    etpd = estimate_total_polynomial_degree

    P2 = FiniteElement("CG", triangle, 2)
    V2 = VectorElement("CG", triangle, 2)
    domain = Mesh(VectorElement("Lagrange", triangle, 1))

    u = Coefficient(FunctionSpace(domain, P2))
    v = Coefficient(FunctionSpace(domain, V2))

    assert etpd(u.dx(0)) == 2 - 1
    assert etpd(grad(u)) == 2 - 1
    assert etpd(nabla_grad(u)) == 2 - 1
    assert etpd(div(u)) == 2 - 1

    assert etpd(v.dx(0)) == 2 - 1
    assert etpd(grad(v)) == 2 - 1
    assert etpd(nabla_grad(v)) == 2 - 1
    assert etpd(div(v)) == 2 - 1
    assert etpd(nabla_div(v)) == 2 - 1

    assert etpd(dot(v, v)) == 2 + 2
    assert etpd(inner(v, v)) == 2 + 2

    assert etpd(dot(grad(u), grad(u))) == 2 - 1 + 2 - 1
    assert etpd(inner(grad(u), grad(u))) == 2 - 1 + 2 - 1

    assert etpd(dot(grad(v), grad(v))) == 2 - 1 + 2 - 1
    assert etpd(inner(grad(v), grad(v))) == 2 - 1 + 2 - 1
