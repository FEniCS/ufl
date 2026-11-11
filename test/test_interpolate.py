"""Test Interpolate object."""

__authors__ = "Nacime Bouziani"
__date__ = "2021-11-19"

import pytest
from utils import FiniteElement, LagrangeElement, MixedElement

from ufl import (
    Action,
    Adjoint,
    Argument,
    Coargument,
    Coefficient,
    Cofunction,
    Constant,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TestFunctions,
    TrialFunction,
    TrialFunctions,
    action,
    adjoint,
    conj,
    derivative,
    dx,
    exp,
    grad,
    inner,
    replace,
    triangle,
)
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.analysis import (
    extract_arguments,
    extract_base_form_operators,
    extract_coefficients,
    extract_terminals_with_domain,
    extract_type,
)
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.expand_indices import expand_indices
from ufl.algorithms.renumbering import renumber_indices
from ufl.algorithms.restructure_base_form import restructure_base_form
from ufl.checks import is_scalar_constant_expression
from ufl.classes import Division, Product, ReferenceGrad, ReferenceValue, Sum
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.form import Form, FormSum, ZeroBaseForm
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1, HInf


@pytest.fixture
def domain_2d():
    return Mesh(LagrangeElement(triangle, 1, (2,)))


@pytest.fixture
def V1(domain_2d):
    f1 = FiniteElement("CG", triangle, 1, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V2(domain_2d):
    f1 = FiniteElement("CG", triangle, 2, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V3(domain_2d):
    f1 = FiniteElement("CG", triangle, 3, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V4(domain_2d):
    f1 = FiniteElement("CG", triangle, 4, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


@pytest.fixture
def V5(domain_2d):
    f1 = FiniteElement("CG", triangle, 5, (), identity_pullback, H1)
    return FunctionSpace(domain_2d, f1)


def test_symbolic(V1, V2):
    u = Coefficient(V1)
    vstar = Argument(V2.dual(), 0)
    Iu = Interpolate(u, vstar)

    assert Iu == Interpolate(u, V2)
    assert Iu.ufl_function_space() == V2
    assert Iu.argument_slots() == (vstar, u)
    assert Iu.arguments() == (vstar,)
    assert Iu.ufl_operands == (u,)
    assert Iu.ufl_element() == V2.ufl_element()
    assert Iu._cache == {}


def test_form_compiler_metadata(domain_2d, V1, V2):
    u = Coefficient(V1)
    cofunction = Cofunction(V2.dual())

    interpolation = Interpolate(u, V2)
    assert interpolation.coefficients() == (u,)
    assert interpolation.ufl_domains() == (domain_2d,)
    assert interpolation.subdomain_data() == {domain_2d: {"cell": [None]}}
    assert interpolation.ufl_element() == V2.ufl_element()

    adjoint_interpolation = Interpolate(TestFunction(V1), cofunction)
    assert adjoint_interpolation.coefficients() == (cofunction,)
    assert adjoint_interpolation.ufl_function_space() == V1.dual()
    assert adjoint_interpolation.ufl_element() == V2.ufl_element()

    scalar_interpolation = Interpolate(u, cofunction)
    assert scalar_interpolation.arguments() == ()
    assert scalar_interpolation.coefficients() == (u, cofunction)
    assert scalar_interpolation.ufl_function_space() is None
    assert scalar_interpolation.ufl_element() == V2.ufl_element()


def test_shape_and_negation(domain_2d, V1, V2):
    scalar_element = V1.ufl_element()
    vector_element = FiniteElement("CG", triangle, 1, (2,), identity_pullback, H1)
    mixed_space = FunctionSpace(domain_2d, MixedElement([scalar_element, vector_element]))
    target_space = FunctionSpace(domain_2d, vector_element)
    _, trial = TrialFunctions(mixed_space)
    _, test = TestFunctions(mixed_space)

    for argument in (trial, test):
        interpolation = Interpolate(argument, target_space)
        assert interpolation.ufl_shape == target_space.value_shape
        assert not isinstance(-interpolation, FormSum)

    assert isinstance(-Interpolate(Coefficient(V1), V2), Product)
    assert isinstance(-Interpolate(Coefficient(V1), Cofunction(V2.dual())), FormSum)


def test_negation_of_contracted_interpolation_with_arguments(V1, V2):
    v = TestFunction(V1)
    interpolation = Interpolate(v, Cofunction(V2.dual()))

    for scaled in (-interpolation, -1 * interpolation):
        assert isinstance(scaled, FormSum)
        assert scaled.arguments() == (v,)
    assert isinstance(inner(1, v) * dx - interpolation, FormSum)


def test_external_operator_follows_dual_slot(V1, V2):
    u = Coefficient(V1)
    v = TestFunction(V1)
    cofunction = Cofunction(V2.dual())
    coargument = Coargument(V2.dual(), 1)

    free = ExternalOperator(u, function_space=V2, argument_slots=(coargument, v))
    assert isinstance(-free, Product)
    assert isinstance(free / 2, Division)

    for contracted in (
        ExternalOperator(u, function_space=V2, argument_slots=(cofunction,)),
        ExternalOperator(u, function_space=V2, argument_slots=(cofunction, v)),
    ):
        assert isinstance(-contracted, FormSum)
        assert isinstance(contracted + contracted, FormSum)
        assert isinstance(contracted - contracted, FormSum)
        assert contracted / 2 == 0.5 * contracted
        with pytest.raises(TypeError):
            contracted / u
        with pytest.raises(TypeError):
            2 / contracted


@pytest.mark.parametrize("operand", ["coefficient", "test_function"])
def test_difference_of_contracted_interpolations_is_a_form_sum(V1, V2, operand):
    expr = Coefficient(V1) if operand == "coefficient" else TestFunction(V1)
    interpolation = Interpolate(expr, Cofunction(V2.dual()))
    linear = inner(1, TestFunction(V1)) * dx if operand == "test_function" else None

    difference = interpolation - interpolation
    assert isinstance(difference, FormSum)
    assert difference.arguments() == interpolation.arguments()
    assert difference.weights() == [1, -1]
    if linear is not None:
        for difference in (interpolation - linear, linear - interpolation):
            assert isinstance(difference, FormSum)
            assert difference.arguments() == interpolation.arguments()


def test_zero_form_has_nothing_to_act_on(V1, V2):
    u = Coefficient(V1)
    zero_form = Interpolate(u, Cofunction(V2.dual()))
    assert isinstance(zero_form * u, Product)
    assert isinstance(u * zero_form, Product)

    with pytest.raises(TypeError):
        (u * dx) * u
    with pytest.raises(ValueError, match="without arguments"):
        action(u * dx, u)
    with pytest.raises(ValueError, match="without arguments"):
        action(zero_form, u)


def test_replace_zero_form(V1, V2):
    u = Coefficient(V1)
    zero_form = Interpolate(u, Cofunction(V2.dual()))
    form = zero_form * u * dx
    assert replace(form, {zero_form: 2.0}) == 2.0 * u * dx


def test_form_multiplied_by_number_scales(domain_2d, V1, V2):
    v = TestFunction(V1)
    u = TrialFunction(V1)
    linear = inner(Coefficient(V1), v) * dx
    constant = Constant(domain_2d)

    assert linear * 2 == 2 * linear
    assert linear * constant == constant * linear
    assert (inner(u, v) * dx * 2).arguments() == (v, u)

    interpolation = Interpolate(v, Cofunction(V2.dual()))
    assert interpolation * 2 == 2 * interpolation


def test_form_acts_on_real_coefficient(domain_2d, V1):
    real_space = FunctionSpace(
        domain_2d, FiniteElement("Real", triangle, 0, (), identity_pullback, HInf)
    )
    v = TestFunction(V1)
    form = inner(TrialFunction(real_space), v) * dx
    assert (form * Coefficient(real_space)).arguments() == (v,)


def test_operator_of_constants_is_constant_only_in_a_real_space(domain_2d, V1, V2):
    real_space = FunctionSpace(
        domain_2d, FiniteElement("Real", triangle, 0, (), identity_pullback, HInf)
    )
    c = Constant(domain_2d)
    for space in (V2, real_space):
        operators = (Interpolate(c, space), ExternalOperator(c, function_space=space))
        for operator in operators:
            assert is_scalar_constant_expression(operator) == (space is real_space)
            assert is_scalar_constant_expression(2 * operator) == (space is real_space)


def test_form_acts_on_operator_of_constants(domain_2d, V1, V2):
    real_space = FunctionSpace(
        domain_2d, FiniteElement("Real", triangle, 0, (), identity_pullback, HInf)
    )
    c = Constant(domain_2d)
    for space in (V1, real_space):
        linear = inner(1, TestFunction(space)) * dx
        for operator in (Interpolate(c, space), ExternalOperator(c, function_space=space)):
            assert linear * operator == action(linear, operator)
            assert (linear * operator).arguments() == ()


@pytest.mark.parametrize("operand", ["coefficient", "constant"])
def test_form_times_zero_form_raises(domain_2d, V1, V2, operand):
    expr = Coefficient(V1) if operand == "coefficient" else Constant(domain_2d)
    zero_form = Interpolate(expr, Cofunction(V2.dual()))
    linear = inner(1, TestFunction(V1)) * dx
    with pytest.raises(ValueError, match="0-form"):
        linear * zero_form
    with pytest.raises(TypeError):
        zero_form * linear


def test_negation_of_interpolation_with_scaled_coargument(V1, V2):
    coargument = Coargument(V2.dual(), 0)
    u = Coefficient(V1)

    for dual_slot in (coargument, 2 * coargument, coargument + coargument):
        assert isinstance(-Interpolate(u, dual_slot), Product)


def test_scaling_agrees_with_negation(domain_2d, V1, V2):
    """Scaling an interpolation must agree with negating it.

    Both follow the dual slot, so that an interpolation of a test function stays
    an expression. Firedrake's fml is
    what notices a disagreement: it builds ``form - label(form)`` and compares
    the result against ``-form``.
    """
    scalar_element = V1.ufl_element()
    vector_element = FiniteElement("CG", triangle, 1, (2,), identity_pullback, H1)
    mixed_space = FunctionSpace(domain_2d, MixedElement([scalar_element, vector_element]))
    target_space = FunctionSpace(domain_2d, vector_element)
    _, trial = TrialFunctions(mixed_space)
    _, test = TestFunctions(mixed_space)

    for argument in (trial, test):
        interpolation = Interpolate(argument, target_space)
        assert type(-interpolation) is type(-1 * interpolation)
        assert type(-interpolation) is type(interpolation * -1)
        assert not isinstance(2 * interpolation, FormSum)
        # Scaling preserves the value shape, which a FormSum would not have.
        assert (2 * interpolation).ufl_shape == target_space.value_shape

    interpolation = Interpolate(Coefficient(V1), V2)
    assert isinstance(-1 * interpolation, Product)
    assert type(-interpolation) is type(-1 * interpolation)

    interpolation = Interpolate(Coefficient(V1), Cofunction(V2.dual()))
    assert isinstance(-1 * interpolation, FormSum)
    assert type(-interpolation) is type(-1 * interpolation)

    # The output of an adjoint interpolation is free, so its sum stays an expression.
    adjoint_interpolation = Interpolate(test, target_space)
    assert isinstance(adjoint_interpolation + adjoint_interpolation, Sum)


def test_form_compiler_signature(V1, V2, V3):
    interpolation = Interpolate(Coefficient(V1), V2)
    equivalent = Interpolate(Coefficient(V1), V2)
    assert interpolation.signature() == equivalent.signature()

    nested = Interpolate(interpolation, V3)
    different_inner_target = Interpolate(Interpolate(Coefficient(V1), V1), V3)
    assert nested.signature() != different_inner_target.signature()

    cofunction_sum = Cofunction(V1.dual()) + Cofunction(V1.dual())
    adjoint_interpolation = Interpolate(TestFunction(V2), cofunction_sum)
    assert isinstance(adjoint_interpolation.signature(), str)


def test_form_compiler_signature_depends_on_interpolation_target(domain_2d):
    """Different target elements must not share an interpolation signature."""
    curl_element = FiniteElement("N1curl", triangle, 1, (2,), identity_pullback, H1)
    lagrange_element = FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1)
    curl_space = FunctionSpace(domain_2d, curl_element)
    lagrange_space = FunctionSpace(domain_2d, lagrange_element)
    x = SpatialCoordinate(domain_2d)

    curl_form = Interpolate(x, curl_space)[0] * dx
    lagrange_form = Interpolate(x, lagrange_space)[0] * dx

    assert curl_form.signature() != lagrange_form.signature()


def test_reference_value_derivative(V1, V2):
    Iu = Interpolate(Coefficient(V1), V2)
    reference_value = ReferenceValue(Iu)

    expression = apply_derivatives(grad(reference_value))
    reference_grads = extract_type(expression, ReferenceGrad)
    assert reference_value in {g.ufl_operands[0] for g in reference_grads}


def test_symbolic_adjoint(V1, V2):
    u = Argument(V1, 0)
    form = inner(1, Argument(V2, 0)) * dx
    cofun = Cofunction(V2.dual())

    for vstar in (form, cofun):
        Iu = Interpolate(u, vstar)

        assert Iu.ufl_function_space() == V1.dual()
        assert Iu.argument_slots() == (vstar, u)
        assert Iu.arguments() == (u,)
        assert Iu.ufl_operands == (u,)


def test_form_dual_slot_argument_is_contracted(V1, V2):
    u = Coefficient(V1)
    v = TestFunction(V2)
    dual_slot = inner(u, v) * dx

    interpolation = Interpolate(u, dual_slot)

    assert interpolation.arguments() == ()


def test_form_dual_slot_contracts_highest_numbered_argument(V1, V2, V3):
    u = Coefficient(V1)
    dual_slot = inner(TrialFunction(V2), TestFunction(V3)) * dx

    interpolation = Interpolate(u, dual_slot)

    assert interpolation.arguments() == (Argument(V3, 0),)


@pytest.mark.parametrize("free, contracted", [(0, 1), (1, 2), (0, 2), (2, 5)])
def test_form_dual_slot_contraction_ignores_numbering_gaps(V1, V2, free, contracted):
    # Both arguments are on V2, so only their numbers tell them apart.
    u = Coefficient(V1)
    dual_slot = Argument(V2, contracted).dx(0) * Argument(V2, free) * dx

    interpolation = Interpolate(u, dual_slot)

    assert interpolation.argument_slots()[0] == dual_slot
    assert interpolation.arguments() == (Argument(V2, free),)


def test_form_dual_slot_with_inner_action(V1, V2):
    # The inner Actions contract Argument(V2, 0), which is not an argument of the slot.
    w = Coefficient(V1)
    inner_action = Action(w, Interpolate(Argument(V1, 0), Coargument(V2.dual(), 1)))
    dual_slot = Action(inner_action, Argument(V2, 1) * Argument(V2, 0) * dx)

    interpolation = Interpolate(Argument(V1, 0), dual_slot)

    assert interpolation.argument_slots()[0] == dual_slot
    assert interpolation.arguments() == (Argument(V1, 0),)


def test_form_dual_slot_numbering_matches_action(V1, V2, V3):
    u = Coefficient(V1)
    A = inner(TrialFunction(V3), TestFunction(V2)) * dx

    interpolation = Interpolate(u, adjoint(A))
    composition = Action(adjoint(A), Interpolate(u, V2))

    assert interpolation.arguments() == composition.arguments() == (Argument(V3, 0),)
    assert len((interpolation + composition).arguments()) == 1


def test_form_dual_slot_arguments_are_replaceable(V1, V2, V3, V4):
    u = Coefficient(V1)
    dual_slot = inner(TrialFunction(V2), TestFunction(V3)) * dx
    interpolation = Interpolate(u, dual_slot)
    w = Argument(V4, 0)

    replaced = replace(interpolation, {interpolation.arguments()[0]: w})

    assert replaced == Interpolate(u, inner(TrialFunction(V2), w) * dx)


@pytest.mark.parametrize("max_rank", [1, 2])
def test_form_dual_slot_contracts_lowest_argument_with_test_function(V1, V2, V3, max_rank):
    v = TestFunction(V1)
    A = inner(TrialFunction(V3), TestFunction(V2)) * dx

    interpolation = Interpolate(v, A)
    composition = Action(Interpolate(v, Argument(V2.dual(), 1)), A)

    assert interpolation.arguments() == composition.arguments() == (v, Argument(V3, 1))

    # v* is numbered after v, so N is the left operand of the Action.
    restructured = restructure_base_form(interpolation, max_composed_rank=max_rank)
    if max_rank == 1:
        expected = composition
    else:
        # A composes with N.
        expected = inner(TrialFunction(V3), Interpolate(v, V2)) * dx
    assert restructured == expected
    assert restructured.arguments() == interpolation.arguments()


@pytest.mark.parametrize("max_rank", [1, 2])
def test_form_dual_slot_rank_two_interpolate(V1, V2, max_rank):
    dual_slot = TestFunction(V2) * dx
    interpolation = Interpolate(TestFunction(V1), dual_slot)

    restructured = restructure_base_form(interpolation, max_composed_rank=max_rank)

    if max_rank == 1:
        expected = interpolation
    else:
        expected = Interpolate(TestFunction(V1), V2) * dx
    assert restructured == expected


def test_form_dual_slot_free_argument_with_expression_number_raises(V1, V2):
    u = Argument(V1, 1)
    dual_slot = Argument(V2, 0) * Argument(V2, 1) * Argument(V2, 2) * dx

    with pytest.raises(ValueError, match=r"Same argument numbers in first and second operands"):
        Interpolate(u, dual_slot)


def test_derivative_in_argument_0_keeps_the_contracted_argument(V1, V2, V3, V4):
    u = Coefficient(V4)
    w = Argument(V1, 0)
    free = Argument(V2, 1)
    dual_slot = inner(Argument(V3, 2), free) * dx

    derivative_form = expand_derivatives(derivative(Interpolate(u, dual_slot), u, w))

    # D_u[w] I(u, F) = I(w, F): the output stays in V3, so V3 stays contracted.
    assert derivative_form == Interpolate(w, dual_slot)
    assert derivative_form.argument_slots()[0] == dual_slot
    assert derivative_form.arguments() == (w, free)


def test_argument_0_contracts_the_highest_argument_of_a_form_without_argument_0(V1, V2, V3):
    w = Argument(V1, 0)
    free = Argument(V2, 1)
    dual_slot = inner(Argument(V3, 2), free) * dx

    assert Interpolate(w, dual_slot).arguments() == (w, free)


@pytest.mark.parametrize(
    "primal_number, free_number, contracted_number, direction_number",
    [(0, 1, 2, 3), (None, 0, 1, 2), (1, 0, 1, 2)],
)
def test_derivative_through_a_form_dual_slot_keeps_the_contracted_argument(
    V1, V2, V3, V4, primal_number, free_number, contracted_number, direction_number
):
    g = Coefficient(V4)
    direction = Argument(V4, direction_number)
    free = Argument(V2, free_number)
    primal = Coefficient(V1) if primal_number is None else Argument(V1, primal_number)
    dual_slot = g * inner(Argument(V3, contracted_number), free) * dx

    derivative_form = expand_derivatives(derivative(Interpolate(primal, dual_slot), g, direction))

    expected = {free, direction} | ({primal} if primal_number is not None else set())
    assert set(derivative_form.arguments()) == expected


def test_form_dual_slot_contracted_argument_can_share_expression_number(V1, V2):
    u = Argument(V1, 1)
    dual_slot = inner(Argument(V2, 1), Argument(V2, 0)) * dx

    interpolation = Interpolate(u, dual_slot)

    assert interpolation.arguments() == (Argument(V2, 0), u)


@pytest.mark.parametrize("max_rank", [1, 2])
def test_adjoint_action_of_form_dual_slot_is_numbered_from_zero(V1, V2, max_rank):
    w = Coefficient(V2)
    A = Argument(V2, 1).dx(0) * Argument(V2, 0) * dx
    N = Interpolate(Argument(V1, 1), A)

    restructured = restructure_base_form(Action(Adjoint(N), w), max_composed_rank=max_rank)

    if max_rank == 1:
        expected = Interpolate(Argument(V1, 0), conj(Argument(V2, 0).dx(0) * w) * dx)
    else:
        expected = conj(Interpolate(Argument(V1, 0), V2).dx(0) * w) * dx
    assert restructured == expected
    assert restructured.arguments() == (Argument(V1, 0),)


def test_form_dual_slot_with_deep_shared_integrand(V1, V2):
    # Each level reuses the previous one twice. A traversal that restarts at every
    # node takes time exponential in the depth.
    u = Coefficient(V1)
    w = Coefficient(V2)
    e = w
    for _ in range(30):
        e = exp(e) + e * w
    dual_slot = e * TestFunction(V2) * dx

    assert Interpolate(u, dual_slot).arguments() == ()


def test_derivative_direction_with_same_argument_number_raises(V1, V2, V3):
    u = Coefficient(V1)
    dual_slot = inner(TrialFunction(V2), TestFunction(V3)) * dx
    dIu = derivative(Interpolate(u, dual_slot), u, Argument(V1, 0))

    with pytest.raises(ValueError, match=r"Same argument numbers in the operator"):
        dIu.arguments()


def test_derivative_direction_with_coargument_number_raises(V1, V2):
    u = Coefficient(V1)
    dIu = derivative(Interpolate(u**2, V2), u, Argument(V1, 0))

    with pytest.raises(ValueError, match=r"Same argument numbers in the operator"):
        expand_derivatives(dIu)


def test_derivative_with_gapped_coargument(V1, V2):
    u = Coefficient(V1)
    J = Interpolate(u, Argument(V2.dual(), 1)) * dx

    dJdu = expand_derivatives(derivative(J, u))

    assert dJdu.arguments() == (Argument(V1, 0),)


def test_derivative_through_a_cofunction_dual_slot_in_an_integrand(V1, V2):
    u = Coefficient(V1)
    v = Argument(V1, 0)
    c = Cofunction(V2.dual())

    dF = expand_derivatives(derivative(Interpolate(u, c) * v * dx, c))

    assert dF == Interpolate(u, Coargument(V2.dual(), 1)) * v * dx


def test_derivative_of_a_form_independent_of_a_cofunction(V1, V2):
    v = Argument(V1, 0)
    c = Cofunction(V2.dual())

    dF = expand_derivatives(derivative(Coefficient(V1) * v * dx, c))

    assert dF == ZeroBaseForm((v, Coargument(V2.dual(), 1)))


def test_derivative_of_a_form_sum_by_a_cofunction(V1, V2):
    u = Coefficient(V1)
    c = Cofunction(V2.dual())

    dF = expand_derivatives(derivative(Interpolate(u, c) + u * dx, c))

    assert dF == Interpolate(u, Coargument(V2.dual(), 0))


@pytest.mark.parametrize("max_rank", [1, 2])
def test_action_adjoint(V1, V2, max_rank):
    # Set dual of V2
    V2_dual = V2.dual()
    vstar = Argument(V2_dual, 0)

    u = Coefficient(V1)
    Iu = Interpolate(u, vstar)

    v1 = TrialFunction(V1)
    Iv = Interpolate(v1, vstar)  # V1 -> V2

    assert Iv.argument_slots() == (vstar, v1)
    assert Iv.arguments() == (vstar, v1)

    # -- Action -- #
    v = TestFunction(V1)
    v2 = TrialFunction(V2)
    F = v2 * v * dx
    assert action(Iv, u) == Action(Iv, u)
    assert action(F, Iv) == Action(F, Iv)
    assert action(F, Iu) == Iu * v * dx

    # -- Adjoint -- #
    assert adjoint(Iv) == Adjoint(Iv)

    # action of one-form on interpolation operator
    one_form = Argument(V2, 0) * dx
    one_form_action = action(one_form, Iv)  # V2^* -> V1^*
    action_one_form = restructure_base_form(one_form_action, max_composed_rank=max_rank)
    if max_rank == 1:
        expected = Interpolate(Argument(V1, 0), one_form)
    else:
        expected = Interpolate(Argument(V1, 0), V2) * dx
    assert action_one_form == expected

    # zero-form case
    action_zero_form = action(one_form, Iu)  # a number
    assert isinstance(action_zero_form, Form)
    assert action_zero_form.arguments() == ()


@pytest.mark.parametrize("max_rank", [1, 2])
def test_differentiation(V1, V2, max_rank):
    u = Coefficient(V1)
    v = TestFunction(V1)

    # Define Interpolate
    Iu = Interpolate(u, V2)

    # -- Differentiate: Interpolate(u, V2) -- #
    uhat = TrialFunction(V1)
    dIu = expand_derivatives(derivative(Iu, u, uhat))

    # dInterpolate(u, v*)/du[uhat] <==> Interpolate(uhat, v*)
    assert dIu == Interpolate(uhat, V2)

    # -- Differentiate: Interpolate(u**2, V2) -- #
    g = u**2
    Ig = Interpolate(g, V2)
    dIg = expand_derivatives(derivative(Ig, u, uhat))
    assert dIg == Interpolate(2 * uhat * u, V2)

    # -- Differentiate: I(u, V2) * v * dx -- #
    F = Iu * v * dx
    Ihat = TrialFunction(Iu.ufl_function_space())
    dFdu = expand_derivatives(derivative(F, u, uhat))
    assert dFdu == dIu * v * dx
    # Restructured: dF/du[uhat] = Action(dF/dIu, dIu/du[uhat])
    dFdIu = expand_derivatives(derivative(F, Iu, Ihat))
    assert dFdIu == Ihat * v * dx
    assert restructure_base_form(dFdu) == Action(dFdIu, dIu)

    # -- Differentiate: u * I(u, V2) * v * dx -- #
    F = u * Iu * v * dx
    dFdu = expand_derivatives(derivative(F, u, uhat))
    assert dFdu == (uhat * Iu + u * dIu) * v * dx
    # Restructured: dF/du[uhat] = ∂F/∂u[uhat] + Action(∂F/∂Iu, dIu/du[uhat])
    expected = uhat * Iu * v * dx + Action(Ihat * u * v * dx, dIu)
    assert restructure_base_form(dFdu) == expected

    # -- Differentiate (wrt Iu): <Iu, v> + <grad(Iu), grad(v)> - <f, v>
    f = Coefficient(V1)
    F = inner(Iu, v) * dx + inner(grad(Iu), grad(v)) * dx - inner(f, v) * dx
    dFdIu = expand_derivatives(derivative(F, Iu, Ihat))
    dFdu = expand_derivatives(derivative(F, u, uhat))
    restructured = restructure_base_form(dFdu)
    assert renumber_indices(restructured) == renumber_indices(Action(dFdIu, dIu))

    # BaseFormOperators are treated as coefficients when a form is differentiated wrt them.
    # -> dFdIu <=> dFdw
    w = Coefficient(V2)
    F = replace(F, {Iu: w})
    dFdw = expand_derivatives(derivative(F, w, Ihat))

    # Need to expand indices to be able to match equal (different MultiIndex used for both).
    assert expand_indices(dFdIu) == expand_indices(dFdw)

    # Derivative of form I(u, V2) wrt coefficient u
    J = Iu * dx
    dJdu = expand_derivatives(derivative(J, u))
    # dJ/du[v0] = Action(I(v0; vhat), dJ/dIu)
    restructured = restructure_base_form(dJdu, max_composed_rank=max_rank)
    if max_rank == 1:
        expected = Interpolate(Argument(V1, 0), Argument(V2, 0) * dx)
    else:
        # dJ/dIu composes with the operator.
        expected = Interpolate(Argument(V1, 0), V2) * dx
    assert restructured == expected
    assert dJdu.arguments() == (Argument(V1, 0),)


def test_formsum_derivative(V1, V2):
    u = Coefficient(V1)
    J = Interpolate(u, V2) ** 2 * dx
    dJ = derivative(J, u)
    # The Cofunction is not under a derivative, so the chain rule only
    # differentiates the interpolation in `dJ`.
    c = Cofunction(V1.dual())
    assert expand_derivatives(FormSum((dJ, 1), (c, 1))) == FormSum(
        (expand_derivatives(dJ), 1), (c, 1)
    )


@pytest.mark.parametrize("max_rank", [1, 2])
def test_second_derivative(V1, V2, max_rank):
    u = Coefficient(V1)
    f = Coefficient(V2)
    Iu = Interpolate(u, V2)
    v0, v1 = TestFunction(V1), TrialFunction(V1)
    w0, w1 = TestFunction(V2), TrialFunction(V2)

    # -- Differentiate: J = 0.5 * (Iu - f)**2 * dx -- #
    J = 0.5 * (Iu - f) ** 2 * dx
    dJdu = expand_derivatives(derivative(J, u))
    # dJ/du[v0] = I^T dJ/dIu, the adjoint interpolation of dJ/dIu
    dJdIu = expand_derivatives(derivative(J, Iu, w0))
    restructured = restructure_base_form(dJdu, max_composed_rank=max_rank)
    if max_rank == 1:
        expected = Interpolate(v0, dJdIu)
    else:
        expected = 0.5 * ((Iu - f) * (2 * Interpolate(v0, V2))) * dx
    assert restructured == expected

    # d2J/du2[v0, v1] = I^T d2J/dIu2 I, since I is linear
    d2JdIu2 = expand_derivatives(derivative(dJdIu, Iu, w1))
    d2Jdu2 = expand_derivatives(derivative(dJdu, u, v1))
    assert d2Jdu2 == 0.5 * (2 * Interpolate(v0, V2) * Interpolate(v1, V2)) * dx
    assert d2Jdu2.arguments() == (v0, v1)
    # Restructured: d2J/du2 = Action(I^T, Action(d2J/dIu2, I))
    vhat = Argument(V2.dual(), 1)
    expected = Action(Interpolate(v0, vhat), Action(d2JdIu2, Interpolate(v1, V2)))
    assert restructure_base_form(d2Jdu2) == expected

    # -- Nested derivatives expand one at a time from the inside out -- #
    H = derivative(derivative(J, u), u)
    assert expand_derivatives(H) == d2Jdu2

    # -- Adjoint interpolation whose dual slot doesn't depend on u -- #
    assert expand_derivatives(derivative(Interpolate(v0, f * w0 * dx), u, v1)) == ZeroBaseForm(
        (v0, v1)
    )

    # -- Nested derivatives with an integral that doesn't depend on Iu -- #
    R = 0.5 * u**2 * dx
    d2Rdu2 = expand_derivatives(derivative(derivative(R, u), u))
    d2JRdu2 = expand_derivatives(derivative(derivative(J + R, u), u))
    assert d2JRdu2 == d2Jdu2 + d2Rdu2


def test_derivative_keeps_interpolate(V1, V2):
    # Subclasses of Interpolate carry data that reconstruction may drop.
    Iu = Interpolate(Coefficient(V1), V2)
    assert apply_derivatives(Iu) is Iu


def test_coefficient_derivatives(V1, V2):
    u = Coefficient(V1)
    g = Coefficient(V1)
    v = TestFunction(V1)
    uhat = TrialFunction(V1)
    Iu = Interpolate(u, V2)

    # The given derivative of Iu replaces the derivative of its operand.
    dF = derivative(Iu * v * dx, u, uhat, coefficient_derivatives={Iu: g})
    assert expand_derivatives(dF) == g * uhat * v * dx


def test_scaled_second_derivative(V1, V2):
    u = Coefficient(V1)
    f = Coefficient(V2)
    Iu = Interpolate(u, V2)
    v0, v1 = TestFunction(V1), TrialFunction(V1)
    w0, w1 = TestFunction(V2), TrialFunction(V2)
    J = 0.5 * (Iu - f) ** 2 * dx
    d2JdIu2 = expand_derivatives(derivative(derivative(J, Iu, w0), Iu, w1))
    H = expand_derivatives(2 * derivative(derivative(J, u), u))
    assert H == 2 * (0.5 * (2 * Interpolate(v0, V2) * Interpolate(v1, V2))) * dx

    # Restructured: 2 d2J/du2[v0, v1] = I^T (2 d2J/dIu2) I
    vhat = Argument(V2.dual(), 1)
    expected = Action(Interpolate(v0, vhat), Action(2 * d2JdIu2, Interpolate(v1, V2)))
    assert restructure_base_form(H) == expected


def test_second_derivative_with_integral_without_interpolate(V1, V2):
    u = Coefficient(V1)
    f = Coefficient(V2)
    Iu = Interpolate(u, V2)
    v0, v1 = TestFunction(V1), TrialFunction(V1)
    w0, w1 = TestFunction(V2), TrialFunction(V2)
    J = 0.5 * (Iu - f) ** 2 * dx
    R = 0.5 * inner(grad(u), grad(u)) * dx
    d2JdIu2 = expand_derivatives(derivative(derivative(J, Iu, w0), Iu, w1))
    d2Rdu2 = expand_derivatives(derivative(derivative(R, u), u))
    H = expand_derivatives(derivative(derivative(J + R, u), u))

    # Moving out one Interpolate also removes the other, which leaves only R.
    vhat = Argument(V2.dual(), 1)
    expected = Action(Interpolate(v0, vhat), Action(d2JdIu2, Interpolate(v1, V2)))
    restructured_R, restructured_J = restructure_base_form(H).components()
    assert renumber_indices(restructured_R) == renumber_indices(d2Rdu2)
    assert restructured_J == expected


def test_dual_slot_derivative_with_coefficient_direction(V1, V2):
    u = Coefficient(V1)
    du = Coefficient(V1)
    v = TestFunction(V2)
    vstar = inner(u, v) * dx
    Iu = Interpolate(u, vstar)

    actual = restructure_base_form(expand_derivatives(derivative(Iu, u, du)))
    expected = inner(u, Interpolate(du, V2)) * dx + inner(du, Interpolate(u, V2)) * dx

    assert actual == expected

    # The coargument in the dual slot does not depend on u.
    assert expand_derivatives(derivative(Interpolate(u, V2), u, du)) == Interpolate(du, V2)


def test_second_derivative_through_nested_formsum(V1, V2):
    u = Coefficient(V1)
    du = Coefficient(V1)
    v = TestFunction(V2)
    Iu = Interpolate(u, inner(u, v) * dx)
    term = inner(du, Interpolate(du, V2)) * dx

    d2Idu2 = restructure_base_form(expand_derivatives(derivative(derivative(Iu, u, du), u, du)))

    # The first derivative is a FormSum because both the operand and dual slot
    # depend on u. Differentiating that nested FormSum gives the product rule twice.
    assert d2Idu2 == term + term


def test_extract_base_form_operators(V1, V2):
    u = Coefficient(V1)
    uhat = TrialFunction(V1)
    vstar = Argument(V2.dual(), 0)

    # -- Interpolate(u, V2) -- #
    Iu = Interpolate(u, V2)
    assert extract_arguments(Iu) == [vstar]
    assert extract_terminals_with_domain(Iu) == ([vstar], [u], [])

    F = Iu * dx
    # Form composition: Iu * dx <=> Action(v * dx, Iu(u; v*))
    assert extract_arguments(F) == []
    assert extract_terminals_with_domain(F) == ([], [u], [])

    for e in [Iu, F]:
        assert extract_coefficients(e) == [u]
        assert extract_base_form_operators(e) == [Iu]

    # -- Interpolate(u, V2) -- #
    Iv = Interpolate(uhat, V2)
    assert extract_arguments(Iv) == [vstar, uhat]
    assert extract_terminals_with_domain(Iv) == ([vstar, uhat], [], [])
    assert extract_coefficients(Iv) == []
    assert extract_base_form_operators(Iv) == [Iv]

    # -- Action(v * v2 * dx, Iv) -- #
    v2 = TrialFunction(V2)
    v = TestFunction(V1)
    F = Action(v * v2 * dx, Iv)
    assert extract_arguments(F) == [v, uhat]


def test_operator_derivative_reconstruction(V1, V2):
    u = Coefficient(V1)

    # Define Interpolate
    Iu = Interpolate(u, V2)

    # -- Differentiate: Interpolate(u, V2) -- #
    uhat = TrialFunction(V1)
    dIu = derivative(Iu, u, uhat)
    # Check operator derivative can be reconstructed
    dIu_r = dIu._ufl_expr_reconstruct_(*dIu.ufl_operands)
    dIu = expand_derivatives(dIu_r)

    assert dIu == Interpolate(uhat, V2)


def test_operator_derivative_equality(V1, V2):
    u = Coefficient(V1)
    uhat = TrialFunction(V1)
    vhat = TestFunction(V1)

    dIu = derivative(Interpolate(u, V2), u, uhat)
    assert dIu == derivative(Interpolate(u, V2), u, uhat)
    assert not dIu == derivative(Interpolate(u, V2), u, vhat)


def test_interpolate_argument_numbering(V1, V2):
    u = Coefficient(V1)
    u0 = Argument(V1, 0)
    u1 = Argument(V1, 1)
    vstar0 = Argument(V2.dual(), 0)
    vstar1 = Argument(V2.dual(), 1)
    cofunc = Cofunction(V2.dual())
    one_form = Argument(V2, 0) * dx

    Interpolate(u, cofunc)
    Interpolate(u, vstar0)
    Interpolate(u, V2)
    Interpolate(u0, cofunc)  # adjoint
    Interpolate(u0, one_form)  # adjoint
    Interpolate(u1, vstar0)
    Interpolate(u0, vstar1)  # adjoint

    with pytest.raises(ValueError, match=r"Same argument numbers in first and second operands"):
        Interpolate(u0, vstar0)

    # Arguments need not be numbered contiguously.
    for interpolation in (
        Interpolate(u, vstar1),
        Interpolate(u1, cofunc),
        Interpolate(u1, one_form),
    ):
        assert tuple(argument.number() for argument in interpolation.arguments()) == (1,)

    u2 = u0 * u1
    with pytest.raises(
        ValueError, match=r"Same argument numbers in first and second operands to interpolate."
    ):
        Interpolate(u2, vstar0)

    with pytest.raises(ValueError, match=r"Expecting a primal function space."):
        Interpolate(u, V2.dual())

    with pytest.raises(
        ValueError, match=r"Expecting the second argument to be FunctionSpace or BaseForm."
    ):
        Interpolate(u, u0)


def test_interpolate_composition(V1, V2, V3, V4, V5):
    u5 = Coefficient(V5)
    u4 = Interpolate(u5, V4)
    assert u4.arguments() == (Argument(V4.dual(), 0),)
    u3 = Interpolate(u4, V3)
    assert u3.arguments() == (Argument(V3.dual(), 0),)
    u2 = Interpolate(u3, V2)
    u1 = Interpolate(u2, V1)

    assert u4.ufl_function_space() == V4
    assert u3.ufl_function_space() == V3
    assert u2.ufl_function_space() == V2
    assert u1.ufl_function_space() == V1

    args = extract_arguments(u1)
    assert set(args) == {
        Argument(V4.dual(), 0),
        Argument(V3.dual(), 0),
        Argument(V2.dual(), 0),
        Argument(V1.dual(), 0),
    }

    assert u1.arguments() == (Argument(V1.dual(), 0),)

    # adjoint
    u1 = Cofunction(V1.dual())
    u2 = Interpolate(Argument(V2, 0), u1)  # V1^* x V2 -> R, equiv V1^* -> V2^*
    u3 = Interpolate(Argument(V3, 0), u2)
    u4 = Interpolate(Argument(V4, 0), u3)
    u5 = Interpolate(Argument(V5, 0), u4)

    assert u2.ufl_function_space() == V2.dual()
    assert u3.ufl_function_space() == V3.dual()
    assert u4.ufl_function_space() == V4.dual()
    assert u5.ufl_function_space() == V5.dual()

    args = extract_arguments(u5)
    assert set(args) == {
        Argument(V5, 0),
        Argument(V4, 0),
        Argument(V3, 0),
        Argument(V2, 0),
        Argument(V1, 0),
    }
    assert u5.arguments() == (Argument(V5, 0),)

    forward_twoform = Interpolate(Argument(V2, 1), V1)  # V2 x V1^* -> R, equiv V2 -> V1
    assert forward_twoform.ufl_function_space() == V1
    assert forward_twoform.arguments() == (Argument(V1.dual(), 0), Argument(V2, 1))
    Iu = Interpolate(forward_twoform, V3)  # V2 x V3^* -> R, equiv V2 -> V3
    assert Iu.ufl_function_space() == V3
    assert Iu.arguments() == (Argument(V3.dual(), 0), Argument(V2, 1))
    assert set(extract_arguments(Iu)) == {
        Argument(V2, 1),
        Argument(V3.dual(), 0),
        Argument(V1.dual(), 0),
    }

    adjoint_twoform = Interpolate(
        Argument(V2, 0), Argument(V1.dual(), 1)
    )  # V1^* x V2 -> R, equiv V1^* -> V2^*
    assert adjoint_twoform.ufl_function_space() == V2.dual()
    assert adjoint_twoform.arguments() == (Argument(V2, 0), Argument(V1.dual(), 1))
    Iu_adj = Interpolate(Argument(V3, 0), adjoint_twoform)  # V1^* x V3 -> R, equiv V1^* -> V3^*
    assert Iu_adj.ufl_function_space() == V3.dual()
    assert Iu_adj.arguments() == (Argument(V3, 0), Argument(V1.dual(), 1))
    assert set(extract_arguments(Iu_adj)) == {
        Argument(V3, 0),
        Argument(V2, 0),
        Argument(V1.dual(), 1),
    }

    with pytest.raises(ValueError, match="Expecting a primal expression to interpolate"):
        Interpolate(u2, V3)


def test_interpolate_expr(V1, V2, V3, V4):
    u4 = Coefficient(V4)
    u31 = Interpolate(u4, V3)
    u32 = Interpolate(u4, V3)
    u3 = u31 + u32
    u2 = Interpolate(u3, V2)
    u1 = Interpolate(u2, V1)

    assert extract_arguments(u3) == []
    assert u1.ufl_function_space() == V1
    assert set(extract_arguments(u1)) == {
        Argument(V3.dual(), 0),
        Argument(V2.dual(), 0),
        Argument(V1.dual(), 0),
    }
    assert u1.arguments() == (Argument(V1.dual(), 0),)
    assert u2.arguments() == (Argument(V2.dual(), 0),)

    # product
    u33 = u31 * u32
    assert extract_arguments(u33) == []
    u21 = Interpolate(u33, V2)
    u11 = Interpolate(u21, V1)
    assert u11.arguments() == (Argument(V1.dual(), 0),)
    assert u21.arguments() == (Argument(V2.dual(), 0),)

    # with MathFunction
    u34 = exp(u31 * u32) + u31 - 1
    assert extract_arguments(u34) == []
    u22 = Interpolate(u34, V2)
    u12 = Interpolate(u22, V1)
    assert u12.arguments() == (Argument(V1.dual(), 0),)
    assert u22.arguments() == (Argument(V2.dual(), 0),)

    # adjoint
    u1 = Cofunction(V1.dual())
    u21 = Interpolate(Argument(V2, 0), u1)
    u22 = Interpolate(Argument(V2, 0), u1)
    u2 = u21 + u22
    u3 = Interpolate(Argument(V3, 0), u2)
    u4 = Interpolate(Argument(V4, 0), u3)

    assert u1.arguments() == (Argument(V1, 0),)
    assert extract_arguments(u2) == [Argument(V2, 0)]
    assert u4.ufl_function_space() == V4.dual()
    assert set(extract_arguments(u4)) == {Argument(V2, 0), Argument(V3, 0), Argument(V4, 0)}
    assert u4.arguments() == (Argument(V4, 0),)
    assert u3.arguments() == (Argument(V3, 0),)


def test_interpolate_cofunction(V1, V2):
    V1_cofunc = Cofunction(V1.dual())
    V2_test = TestFunction(V2)

    Iu_adj = Interpolate(V2_test, V1_cofunc)  # Cofunction in V2^*
    assert Iu_adj.arguments() == (Argument(V2, 0),)

    # Test zero-form
    V1_coeff = Coefficient(V1)
    V2_coeff = Interpolate(V1_coeff, V2)
    Iu_zeroform = Interpolate(V2_coeff, V1_cofunc)
    assert Iu_zeroform.arguments() == ()

    Iu2 = Interpolate(Iu_zeroform * V1_coeff, V2)
    assert Iu2.arguments() == (Argument(V2.dual(), 0),)

    Iu3 = V1_coeff * Iu_zeroform
    assert extract_arguments(Iu3) == []


def test_interpolate_form(V1, V2):
    V1_coeff = Coefficient(V1)
    V1_test = TestFunction(V1)
    V1_trial = TrialFunction(V1)

    Iu = Interpolate(V1_coeff, V2)
    F = V1_test * Iu * dx

    assert extract_arguments(Iu) == [Argument(V2.dual(), 0)]
    assert extract_arguments(F) == [V1_test]
    assert extract_terminals_with_domain(F) == ([V1_test], [V1_coeff], [])

    # Forward interpolation moves trial function from V1 to V2
    V2_trial = TrialFunction(V2)
    Iu = Interpolate(V2_trial, V1)  # V2 x V1^* -> R, equiv V2 -> V1
    F = Iu * V1_test * dx
    assert extract_arguments(Iu) == [Argument(V1.dual(), 0), V2_trial]
    assert extract_arguments(F) == [V1_test, V2_trial]
    assert F.arguments() == (V1_test, V2_trial)

    F_act = V1_trial * V1_test * dx  # V1 x V1 -> R, equiv V1 -> V1^*
    assert F_act.arguments() == (V1_test, V1_trial)
    G = action(F_act, Iu)  # V2 -> V1 -> V1^*, equiv V2 x V1 -> R
    assert extract_arguments(G) == [V1_test, V2_trial]
    assert G.arguments() == (V1_test, V2_trial)

    # Adjoint interpolation moves test function from V1 to V2
    V2_test = TestFunction(V2)
    Iu_adj = Interpolate(V2_test, V1)  # V1^* x V2 -> R, equiv V1^* -> V2^*
    assert extract_arguments(Iu_adj) == [Argument(V2, 0), Argument(V1.dual(), 1)]
    assert Iu_adj.arguments() == (Argument(V2, 0), Argument(V1.dual(), 1))
    G2 = action(Iu_adj, F_act)  # V1 -> V1^* -> V2^*, equiv V1 x V2 -> R
    assert extract_arguments(G2) == [V2_test, V1_trial]
    assert G2.arguments() == (V2_test, V1_trial)

    # alternatively using adjoint(Iu)
    Iu_adj = adjoint(Iu)
    G2 = action(Iu_adj, F_act)
    assert extract_arguments(G2) == [V2_test, V1_trial]
    assert G2.arguments() == (V2_test, V1_trial)

    # Adjoint interpolation of Form
    G3 = Interpolate(V2_test, F_act)
    assert G3.arguments() == (V2_test, V1_trial)
    assert G3.ufl_function_space() == V2.dual()


def test_interpolate_adjoint(V1, V2):
    V1_trial = TrialFunction(V1)
    Iu_forward = Interpolate(V1_trial, V2)  # V1 x V2^* -> R, equiv V1 -> V2
    Iu_adjoint = adjoint(Iu_forward)  # V2^* x V1 -> R, equiv V2^* -> V1^*

    assert Iu_forward.arguments() == (Argument(V2.dual(), 0), V1_trial)
    assert Iu_forward.ufl_function_space() == V2
    assert Iu_adjoint.arguments() == (Argument(V1, 0), Argument(V2.dual(), 1))

    Iu_adjoint2 = Interpolate(Argument(V1, 0), Argument(V2.dual(), 1))
    assert Iu_adjoint2.arguments() == Iu_adjoint.arguments()

    Iu = adjoint(Iu_adjoint)
    assert Iu.arguments() == Iu_forward.arguments()


def test_interpolate_formsum(V1, V2):
    V1_cofunc = Cofunction(V1.dual())
    V1_cofunc2 = Cofunction(V1.dual())
    V1_sum = V1_cofunc + V1_cofunc2
    assert isinstance(V1_sum, FormSum)
    assert V1_sum.arguments() == (Argument(V1, 0),)

    V2_cofunc = Interpolate(Argument(V2, 0), V1_sum)  # Cofunction in V2^*
    assert V2_cofunc.arguments() == (Argument(V2, 0),)
    assert V2_cofunc.ufl_function_space() == V2.dual()


def test_equal_derivatives_of_equal_forms_with_interpolate(V1, V2):
    u = Coefficient(V1)
    v = TestFunction(V2)
    F1 = inner(Interpolate(u, V2), v) * dx
    F2 = inner(Interpolate(u, V2), v) * dx
    assert F1 is not F2

    assert bool(derivative(F1, u) == derivative(F2, u))
    assert not bool(derivative(F1, u) == derivative(2 * F2, u))


@pytest.mark.parametrize("number", [0, 5])
def test_derivative_dual_slot_with_direction_numbered_zero(V1, V2, number):
    u = Coefficient(V1)
    v = TestFunction(V1)
    N = Interpolate(u**2, inner(u, Argument(V2, number)) * dx)

    dN = expand_derivatives(derivative(N, u, v))
    assert dN.arguments() == (v,)


def test_second_derivative_dual_slot_with_directions_numbered_from_zero(V1, V2):
    u = Coefficient(V1)
    N = Interpolate(u**2, inner(u, TestFunction(V2)) * dx)
    v0, v1 = Argument(V1, 0), Argument(V1, 1)

    ddN = expand_derivatives(derivative(derivative(N, u, v0), u, v1))
    assert ddN.arguments() == (v0, v1)


def test_derivative_default_direction_skips_contracted_argument(V1, V2):
    u = Coefficient(V1)
    N = Interpolate(u**2, inner(u, TestFunction(V2)) * dx)

    dN = expand_derivatives(derivative(N, u))
    assert [a.number() for a in dN.arguments()] == [0]
    assert all([a.number() for a in component.arguments()] == [0] for component in dN.components())

    ddN = expand_derivatives(derivative(derivative(N, u), u))
    assert [a.number() for a in ddN.arguments()] == [0, 1]


def test_derivative_default_direction_after_free_argument(V1, V2):
    u = Coefficient(V1)
    Iu = Interpolate(u, V2)
    assert [a.number() for a in Iu.arguments()] == [0]

    dIu = expand_derivatives(derivative(Iu, u))
    assert [a.number() for a in dIu.arguments()] == [0, 1]


def test_derivative_of_form_sum_drops_vanishing_component(V1, V2):
    u = Coefficient(V1)
    v = TestFunction(V1)
    F = action(Action(inner(TrialFunction(V2), v) * dx, Interpolate(TrialFunction(V1), V2)), u)
    L = v * dx

    dF = expand_derivatives(derivative(F, u))
    for G in (F - L, -L + F):
        J = expand_derivatives(derivative(G, u))
        assert not isinstance(J, FormSum)
        assert J.arguments() == (v, Argument(V1, 1))
        assert J == dF


def test_derivative_of_form_sum_keeps_arguments_when_all_components_vanish(V1):
    u = Coefficient(V1)
    v = TestFunction(V1)
    w = Cofunction(V1.dual())

    J = expand_derivatives(derivative(v * dx + w, u))
    assert isinstance(J, ZeroBaseForm)
    assert J.arguments() == (v, Argument(V1, 1))


@pytest.mark.parametrize("scale", [1, 2])
def test_action_leaves_coargument_of_interpolate_operand_bound(V1, scale):
    # <ustar, I(scale * I(w))> pairs ustar with the output of the outer Interpolate only.
    w = Coefficient(V1)
    ustar = Cofunction(V1.dual())
    operand = Interpolate(w, V1)
    if scale != 1:
        operand = scale * operand

    restructured = restructure_base_form(Action(ustar, Interpolate(operand, V1)))

    assert restructured == Interpolate(operand, ustar)
    assert restructured.arguments() == ()
