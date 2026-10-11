from utils import LagrangeElement

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TrialFunction,
    as_vector,
    derivative,
    ds,
    dx,
    replace,
    triangle,
)
from ufl.algorithms.replace_derivative_nodes import replace_derivative_nodes
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


def test_replace_derivative_nodes_with_equal_coefficient():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u, w = Coefficient(V), Coefficient(V)
    u_copy = Coefficient(V, count=u.count())
    expr = derivative(u**2, u, TrialFunction(V)) + w
    assert _contains(replace_derivative_nodes(expr, {u: u_copy}), u_copy)
