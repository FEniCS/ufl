import pytest
from utils import LagrangeElement

from ufl import (
    Action,
    Coargument,
    Coefficient,
    Cofunction,
    FunctionSpace,
    Identity,
    Matrix,
    Mesh,
    TestFunction,
    TrialFunction,
    as_vector,
    derivative,
    det,
    diff,
    div,
    ds,
    dx,
    grad,
    inner,
    ln,
    replace,
    tr,
    triangle,
    variable,
)
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import has_exact_type
from ufl.constantvalue import Zero
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.corealg.traversal import unique_pre_traversal
from ufl.differentiation import BaseFormDerivative, Div
from ufl.form import Form


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


@pytest.mark.parametrize("with_derivative", [False, True])
def test_replace_preserves_divergence(with_derivative):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(V)
    unew = Coefficient(V)

    expression = div(u)
    expected = div(unew)
    if with_derivative:
        Q = FunctionSpace(domain, LagrangeElement(triangle, 1))
        f = Coefficient(Q)
        h = Coefficient(Q)
        expression += derivative(f**2, f, h)
        expected += 2 * h * f

    assert replace(expression, {u: unew}) == expected


def test_replace_derivative_preserves_divergence():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(V)
    unew = Coefficient(V)
    h = Coefficient(V)
    F = div(u) ** 2 * dx

    result = replace(derivative(F, u, h), {u: unew})

    assert result == 2 * div(h) * div(unew) * dx


def test_replace_derivative_of_variable_derivative():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(V)
    w = Coefficient(V)
    v = TestFunction(V)
    du = TrialFunction(V)

    def jacobian(u):
        F = variable(Identity(2) + grad(u))
        J = det(F)
        psi = 0.5 * (tr(F.T * F) - 2) - ln(J) + 0.5 * ln(J) ** 2
        return derivative(inner(diff(psi, F), grad(v)) * dx, u, du)

    result = replace(jacobian(u), {u: w})

    expected = expand_derivatives(jacobian(w))
    assert expand_derivatives(result).signature() == expected.signature()


def test_replace_keeps_arguments_of_a_vanishing_derivative():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u = Coefficient(V)
    w = Coefficient(V)
    v = TestFunction(V)
    du = TrialFunction(V)
    J = derivative(u * v * dx, w, du)
    assert replace(J, {u: 2 * w}).arguments() == (v, du)


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


@pytest.mark.parametrize("replacement", ["coefficient", "expression", "operator"])
def test_replace_base_form_derivative(V1, replacement):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(N * u * v * dx, u)
    assert isinstance(dJ, BaseFormDerivative)

    replacements = {
        "coefficient": w,
        "expression": 2 * w,
        "operator": ExternalOperator(w, function_space=V1),
    }
    mapping = {u: replacements[replacement]}
    assert replace(dJ, mapping) == replace(expand_derivatives(dJ), mapping)


def test_replace_base_form_derivative_aliasing(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(Action(N * w * v * dx, u), u)

    # The differentiation variable maps to another coefficient of the form,
    # and only the first of the two is differentiated.
    assert replace(dJ, {u: w}) == replace(expand_derivatives(dJ), {u: w})


def test_replace_base_form_derivative_preserves_spatial_derivatives(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(N * div(grad(u)) * v * dx, u)

    forms = [c for c in replace(dJ, {u: w}).components() if isinstance(c, Form)]
    assert forms
    assert all(has_exact_type(form, Div) for form in forms)


def test_replace_leaves_coargument_of_operand_bound(V1):
    # The Coargument of N is bound in the operand of the Interpolate: it pairs with
    # the output of N. Only the Coargument in the dual slot of the Interpolate is free.
    N = ExternalOperator(Coefficient(V1), function_space=V1)
    ustar = Cofunction(V1.dual())
    interpolation = Interpolate(N, V1)
    (vstar,) = interpolation.arguments()

    assert replace(interpolation, {vstar: ustar}) == Interpolate(N, ustar)
