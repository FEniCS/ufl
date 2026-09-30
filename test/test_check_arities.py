import pytest
from utils import LagrangeElement

from ufl import (
    Coefficient,
    FacetNormal,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    adjoint,
    as_tensor,
    cofac,
    conditional,
    conj,
    curl,
    derivative,
    div,
    ds,
    dx,
    grad,
    inner,
    nabla_div,
    nabla_grad,
    tetrahedron,
    triangle,
)
from ufl.algorithms.check_arities import ArityMismatch, check_form_arity, check_integrand_arity
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.classes import ReferenceCurl, ReferenceDiv, ReferenceGrad, ReferenceValue
from ufl.core.interpolate import Interpolate


def test_check_arities():
    # Code from bitbucket issue #49
    cell = tetrahedron
    D = Mesh(LagrangeElement(cell, 1, (3,)))
    V = FunctionSpace(D, LagrangeElement(cell, 2, (3,)))
    dv = TestFunction(V)
    du = TrialFunction(V)

    X = SpatialCoordinate(D)
    N = FacetNormal(D)

    u = Coefficient(V)
    x = X + u
    F = grad(x)
    n = cofac(F) * N

    M = inner(x, n) * ds
    L = derivative(M, u, dv)
    a = derivative(L, u, du)

    compute_form_data(M)
    compute_form_data(L)
    compute_form_data(a)


def test_interpolate_arity():
    domain = Mesh(LagrangeElement(tetrahedron, 1, (3,)))
    V = FunctionSpace(domain, LagrangeElement(tetrahedron, 2))
    v = TestFunction(V)
    u = TrialFunction(V)

    integrand = inner(Interpolate(u, V), v)
    check_integrand_arity(integrand, (v, u))
    check_integrand_arity(integrand, (v, u), complex_mode=True)


def test_complex_arities():
    cell = tetrahedron
    D = Mesh(LagrangeElement(cell, 1, (3,)))
    V = FunctionSpace(D, LagrangeElement(cell, 2, (3,)))
    v = TestFunction(V)
    u = TrialFunction(V)

    # Valid form.
    F = inner(u, v) * dx
    compute_form_data(F, complex_mode=True)
    # Check that adjoint conjugates correctly
    compute_form_data(adjoint(F), complex_mode=True)

    with pytest.raises(ArityMismatch):
        compute_form_data(inner(v, u) * dx, complex_mode=True)

    with pytest.raises(ArityMismatch):
        compute_form_data(inner(conj(v), u) * dx, complex_mode=True)


@pytest.mark.parametrize(
    ("operator", "argument_shape", "result_shape"),
    [
        (grad, (), (2,)),
        (ReferenceGrad, (), (2,)),
        (nabla_grad, (), (2,)),
        (div, (2,), ()),
        (ReferenceDiv, (2,), ()),
        (nabla_div, (2,), ()),
        (curl, (), (2,)),
        (ReferenceCurl, (), (2,)),
    ],
)
def test_complex_arities_of_linear_differential_operators(operator, argument_shape, result_shape):
    """Linear differential operators preserve complex form arity."""
    cell = triangle
    domain = Mesh(LagrangeElement(cell, 1, (2,)))
    argument_space = FunctionSpace(domain, LagrangeElement(cell, 1, argument_shape))
    result_space = FunctionSpace(domain, LagrangeElement(cell, 1, result_shape))
    v = TestFunction(argument_space)
    u = TrialFunction(result_space)

    operand = ReferenceValue(v) if operator in (ReferenceGrad, ReferenceDiv, ReferenceCurl) else v
    form = inner(u, operator(operand)) * dx
    check_form_arity(form, (v, u), complex_mode=True)


def test_product_arity():
    cell = tetrahedron
    D = Mesh(LagrangeElement(cell, 1, (3,)))
    V = FunctionSpace(D, LagrangeElement(cell, 2, (3,)))
    v = TestFunction(V)
    u = TrialFunction(V)

    with pytest.raises(ArityMismatch):
        F = inner(u, u) * dx
        compute_form_data(F, complex_mode=True)

    with pytest.raises(ArityMismatch):
        L = inner(v, v) * dx
        compute_form_data(L, complex_mode=False)


def test_zero_simplify_arity():
    """
    Test that adding verious zero-like expressions to a form is simplified,
    such that one can compute form data for the integral.
    """
    cell = tetrahedron
    D = Mesh(LagrangeElement(cell, 1, (3,)))
    V = FunctionSpace(D, LagrangeElement(cell, 2))
    v = TestFunction(V)
    u = Coefficient(V)

    nonzero = 1
    with pytest.raises(ArityMismatch):
        F = inner(u, v + nonzero) * dx
        compute_form_data(F)
    z = Coefficient(V)

    # Add a Zero-component (rank-0) of a tensor to a rank-1 tensor
    zero = as_tensor([0, z])[0]
    F = inner(u, v + zero) * dx
    fd = compute_form_data(F)
    assert fd.num_coefficients == 1

    # Add a conditional that should have been simplified to zero (rank-0)
    # to a rank-1 tensor
    zero = conditional(z < 0, 0, 0)
    F = inner(u, v + zero) * dx
    fd = compute_form_data(F)
    assert fd.num_coefficients == 1

    # Check that nested zero conditionals are simplifed to zero (rank-0)
    # and can be added to a rank-1 tensor
    zero = conditional(z < 0, 0, conditional(z == 0, 0, 0))
    F = inner(u, v + zero) * dx
    fd = compute_form_data(F)
    assert fd.num_coefficients == 1
