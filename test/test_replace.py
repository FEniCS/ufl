import pytest
from utils import ForeignDerivative, LagrangeElement, MixedElement

from ufl import (
    Action,
    Argument,
    Coargument,
    Coefficient,
    Cofunction,
    FunctionSpace,
    Identity,
    Matrix,
    Mesh,
    TestFunction,
    TestFunctions,
    TrialFunction,
    as_tensor,
    as_vector,
    derivative,
    det,
    diff,
    div,
    ds,
    dx,
    grad,
    indices,
    inner,
    ln,
    replace,
    split,
    sym,
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
from ufl.differentiation import Div
from ufl.form import Form, FormSum, ZeroBaseForm


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


def test_replace_derivative_of_variable_derivative_of_div():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(V)
    w = Coefficient(V)
    v = TestFunction(V)
    du = TrialFunction(V)

    def jacobian(u):
        eps = variable(sym(grad(u)))
        psi = inner(eps, eps) + 0.5 * div(u) ** 2
        return derivative(inner(diff(psi, eps), grad(v)) * dx, u, du)

    result = replace(jacobian(u), {u: w})

    expected = expand_derivatives(jacobian(w))
    assert expand_derivatives(result).signature() == expected.signature()


def test_replace_derivative_of_variable_derivative_of_grad():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 2, (2,)))
    u = Coefficient(V)
    w = Coefficient(V)
    v = TestFunction(V)
    du = TrialFunction(V)

    F = variable(grad(u))
    psi = inner(F, F) ** 2
    i, j, k = indices(3)
    # d/dF grad(psi) = grad(dpsi/dF), with the gradient index placed before those of F
    expected = as_tensor(grad(diff(psi, F))[i, j, k], (k, i, j))

    def jacobian(e):
        return replace(derivative(inner(e, grad(grad(v))) * dx, u, du), {u: w})

    assert jacobian(diff(grad(psi), F)).signature() == jacobian(expected).signature()


def test_replace_keeps_arguments_of_a_vanishing_derivative():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    u = Coefficient(V)
    w = Coefficient(V)
    v = TestFunction(V)
    du = TrialFunction(V)
    J = derivative(u * v * dx, w, du)
    assert replace(J, {u: 2 * w}).arguments() == (v, du)


def test_replace_foreign_derivative():
    """Replacement preserves derivatives supplied by other packages."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    space = FunctionSpace(domain, LagrangeElement(triangle, 1))
    f = Coefficient(space)
    g = Coefficient(space)
    test = TestFunction(space)

    # replace Expr with ForeignDerivative
    expr = derivative(f.dx(0) * f, f) + ForeignDerivative(f.dx(0))
    result = replace(expr, {f: g})
    expected = test.dx(0) * g + g.dx(0) * test + ForeignDerivative(g.dx(0))
    assert result == expected

    # replace Form with ForeignDerivative
    Df = ForeignDerivative(f)
    form = derivative(inner(f.dx(0), f.dx(0)) * dx, f) + inner(Df, test) * dx
    result = replace(form, {f: g})

    Dg = ForeignDerivative(g)
    expected = (inner(g.dx(0), test.dx(0)) + inner(test.dx(0), g.dx(0))) * dx + inner(Dg, test) * dx
    assert result == expected

    # differentiate through a ForeignDerivative independent of f
    trial = TrialFunction(space)
    form = derivative(Dg * f**2 * test * dx, f, trial)
    result = replace(form, {f: 3 * g})
    expected = 2 * trial * (3 * g) * Dg * test * dx
    assert result == expected

    # differentiate through a ForeignDerivative of f
    form = derivative(ForeignDerivative(f**2) * dx, f)
    result = replace(form, {f: g})
    expected = ForeignDerivative(2 * test * g) * dx
    assert result == expected

    # differentiate through a variable derivative of a ForeignDerivative
    x = variable(f)
    form = derivative(diff(ForeignDerivative(x**2), x) * dx, f)
    result = replace(form, {f: g})
    expected = ForeignDerivative(2.0 * test) * dx
    assert result == expected


def test_replace_mixed_element():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    element = MixedElement([LagrangeElement(triangle, 1), LagrangeElement(triangle, 1)])
    space = FunctionSpace(domain, element)
    z = Coefficient(space)
    w = Coefficient(space)

    u, p = split(z)
    form = derivative((inner(grad(u), grad(u)) + inner(p, p)) * dx, z)
    result = replace(form, {z: w})

    u, p = split(w)
    expected = derivative((inner(grad(u), grad(u)) + inner(p, p)) * dx, w)
    expected = expand_derivatives(expected)
    # result and expected are built independently, so their dummy
    # summation indices may be numbered differently; compare signatures
    # rather than raw equality.
    assert expand_derivatives(result).signature() == expected.signature()


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


@pytest.mark.parametrize("operator", ["interpolate", "external_operator"])
def test_replace_primal_slot_by_zero(V1, operator):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    if operator == "interpolate":
        N = Interpolate(w * v, V1)
    else:
        N = ExternalOperator(u, function_space=V1, argument_slots=(Coargument(V1.dual(), 1), w))

    # The operator is linear in its primal slots.
    assert replace(N, {w: Zero()}) == Zero()


@pytest.mark.parametrize("dual_slot", ["cofunction", "form"])
@pytest.mark.parametrize("operator", ["interpolate", "external_operator"])
def test_replace_primal_slot_by_zero_in_form(V1, operator, dual_slot):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    if dual_slot == "cofunction":
        vstar = Cofunction(V1.dual())
        arguments = (v,)
    else:
        vstar = inner(TrialFunction(V1), TestFunction(V1)) * dx
        v1 = Argument(V1, 1)
        arguments = (v, v1)
    if operator == "interpolate":
        N = Interpolate(w * v, vstar)
    else:
        N = ExternalOperator(u, function_space=V1, argument_slots=(vstar, w * v))

    # A contracted dual slot makes the operator a form, so it becomes a zero form
    # with the same arguments.
    assert replace(N, {w: Zero()}) == ZeroBaseForm(arguments)


def test_replace_dual_slot_in_sum(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    vstar = Coargument(V1.dual(), 0)
    c = Cofunction(V1.dual())
    N = ExternalOperator(u, function_space=V1, argument_slots=(vstar,))
    M = ExternalOperator(w, function_space=V1, argument_slots=(vstar,))

    # The contracted operators are forms, so their sum is a FormSum.
    r = replace(N + M, {vstar: c})
    assert isinstance(r, FormSum)
    assert set(r.components()) == {replace(N, {vstar: c}), replace(M, {vstar: c})}


@pytest.mark.parametrize("replacement", ["coefficient", "expression", "operator"])
def test_replace_form_derivative(V1, replacement):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(N * u * v * dx, u)
    assert isinstance(dJ, Form)

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


def test_replace_form_derivative_spatial(V1):
    u = Coefficient(V1)
    w = Coefficient(V1)
    v = TestFunction(V1)
    N = ExternalOperator(u, function_space=V1)
    dJ = derivative(N * div(grad(u)) * v * dx, u)
    assert isinstance(dJ, Form)
    assert has_exact_type(replace(dJ, {u: w}), Div)


def test_replace_leaves_coargument_of_operand_bound(V1):
    # The Coargument of N is bound in the operand of the Interpolate: it pairs with
    # the output of N. Only the Coargument in the dual slot of the Interpolate is free.
    N = ExternalOperator(Coefficient(V1), function_space=V1)
    ustar = Cofunction(V1.dual())
    interpolation = Interpolate(N, V1)
    (vstar,) = interpolation.arguments()

    assert replace(interpolation, {vstar: ustar}) == Interpolate(N, ustar)


def test_replace_interpolate_of_mixed_component(V1):
    # The Interpolate is scalar, although its argument has the shape of the mixed space.
    element = MixedElement([LagrangeElement(triangle, 1), LagrangeElement(triangle, 2)])
    Z = FunctionSpace(V1.ufl_domain(), element)
    _, w = TestFunctions(Z)
    u = Coefficient(V1)
    c = Coefficient(V1)
    form = Interpolate(w, V1) * u * dx

    assert replace(form, {Interpolate(w, V1): c}) == c * u * dx
