import pytest
from utils import LagrangeElement

from ufl import (
    Action,
    Coargument,
    Coefficient,
    Cofunction,
    FunctionSpace,
    Matrix,
    Mesh,
    TestFunction,
    as_vector,
    derivative,
    ds,
    dx,
    inner,
    replace,
    triangle,
)
from ufl.algorithms import expand_derivatives
from ufl.constantvalue import Zero
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.corealg.traversal import unique_pre_traversal
from ufl.differentiation import BaseFormDerivative


def _contains(expr, obj):
    return any(e is obj for e in unique_pre_traversal(expr))


@pytest.fixture
def V1():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    return FunctionSpace(domain, LagrangeElement(triangle, 1))


def test_replace_keeps_mapped_and_unmapped_objects():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u = Coefficient(V)
    w = Coefficient(V)
    w_copy = Coefficient(V, count=w.count())
    first, second = replace(as_vector([w, u]), {u: w_copy}).ufl_operands
    assert first is w
    assert second is w_copy


def test_replace_keeps_equal_objects_of_other_integrals():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u = Coefficient(V)
    w = Coefficient(V)
    z = Coefficient(V)
    w_copy = Coefficient(V, count=w.count())
    F = replace(w * u * dx + w_copy * u * ds, {u: z})
    (ds_integral,) = [itg for itg in F.integrals() if itg.integral_type() == "exterior_facet"]
    assert _contains(ds_integral.integrand(), w_copy)


def test_replace(V1):
    u = Coefficient(V1, count=0)
    N = ExternalOperator(u, function_space=V1)

    # dN(u; uhat, v*)
    dN = expand_derivatives(derivative(N, u))
    vstar, uhat = dN.arguments()
    assert isinstance(vstar, Coargument)

    # Replace v* by a Form
    v = TestFunction(V1)
    F = inner(u, v) * dx
    G = replace(dN, {vstar: F})

    dN_replaced = dN._ufl_expr_reconstruct_(u, argument_slots=(F, uhat))
    assert G == dN_replaced

    # Replace v* by an Action
    M = Matrix(V1, V1)
    A = Action(M, u)
    G = replace(dN, {vstar: A})

    dN_replaced = dN._ufl_expr_reconstruct_(u, argument_slots=(A, uhat))
    assert G == dN_replaced


def test_replace_base_form_operator(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    M = ExternalOperator(w, function_space=V1)
    Iw = Interpolate(w, V1)
    e = N * v + M * v + Iw * v

    # Replacing N by zero drops its term and keeps the operators that don't depend on N.
    # Check them before comparing r, since == shares the operands of equal expressions.
    r = replace(e, {N: Zero()})
    operators = [
        o for o in unique_pre_traversal(r) if isinstance(o, ExternalOperator | Interpolate)
    ]
    assert len(operators) == 2
    assert all(o is M or o is Iw for o in operators)
    assert r == M * v + Iw * v


def test_replace_base_form_derivative(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(Action(N * u * v * dx, u), u)
    assert isinstance(dJ, BaseFormDerivative)

    # The derivative is not applied, and the replacement is applied to the
    # differentiation variable as well as to the base form.
    Nw = ExternalOperator(w, function_space=V1)
    expected = derivative(Action(Nw * w * v * dx, w), w)
    result = replace(dJ, {u: w})
    assert isinstance(result, BaseFormDerivative)
    assert result.ufl_operands == expected.ufl_operands
    assert expand_derivatives(result) == expand_derivatives(expected)


def test_replace_base_form_derivative_aliasing(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(Action(N * w * v * dx, u), u)

    # Mapping the differentiation variable to another coefficient of the
    # form does not commute with the derivative.
    with pytest.raises(ValueError, match="Derivatives should be applied"):
        replace(dJ, {u: w})


def test_replace_leaves_coargument_of_operand_bound(V1):
    # The Coargument of N is bound in the operand of the Interpolate: it pairs with
    # the output of N. Only the Coargument in the dual slot of the Interpolate is free.
    N = ExternalOperator(Coefficient(V1), function_space=V1)
    ustar = Cofunction(V1.dual())
    interpolation = Interpolate(N, V1)
    (vstar,) = interpolation.arguments()

    assert replace(interpolation, {vstar: ustar}) == Interpolate(N, ustar)
