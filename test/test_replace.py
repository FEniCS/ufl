import pytest
from utils import LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    as_vector,
    derivative,
    div,
    ds,
    dx,
    replace,
    triangle,
)
from ufl.corealg.traversal import unique_pre_traversal


def _contains(expr, obj):
    return any(e is obj for e in unique_pre_traversal(expr))


def test_replace_keeps_mapped_and_unmapped_objects():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u, w = Coefficient(V), Coefficient(V)
    w_copy = Coefficient(V, count=w.count())
    first, second = replace(as_vector([w, u]), {u: w_copy}).ufl_operands
    assert first is w
    assert second is w_copy


def test_replace_keeps_equal_objects_of_other_integrals():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u, w, z = Coefficient(V), Coefficient(V), Coefficient(V)
    w_copy = Coefficient(V, count=w.count())
    F = replace(w * u * dx + w_copy * u * ds, {u: z})
    (ds_integral,) = [itg for itg in F.integrals() if itg.integral_type() == "exterior_facet"]
    assert _contains(ds_integral.integrand(), w_copy)


@pytest.mark.parametrize("with_derivative", [False, True])
def test_replace_preserves_divergence(with_derivative):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1, (2,)))
    u, unew = Coefficient(V), Coefficient(V)

    expression, expected = div(u), div(unew)
    if with_derivative:
        Q = FunctionSpace(domain, LagrangeElement(triangle, 1))
        f, h = Coefficient(Q), Coefficient(Q)
        expression += derivative(f**2, f, h)
        expected += 2 * h * f

    assert replace(expression, {u: unew}) == expected


def test_replace_derivative_preserves_divergence():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1, (2,)))
    u, unew, h = Coefficient(V), Coefficient(V), Coefficient(V)
    F = div(u) ** 2 * dx

    result = replace(derivative(F, u, h), {u: unew})

    assert result == 2 * div(h) * div(unew) * dx
