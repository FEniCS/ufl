"""Contraction rules, assemblability and cost specifications for BaseForm restructuring.

For derivative-expanded input, every assembly node must have rank at most
two. A BaseForm used as a value in an integrand, operator operand or argument
slot must have rank at most one. Arguments and Coarguments bound inside a
form or operator stay symbolic. Action operands must have rank one or two,
Adjoint operands must have rank two, and FormSum components must have the
same rank. Assume assembly is implemented for each Form and BaseFormOperator.

For rank-one results, optimality additionally requires no newly assembled
rank-two forms. Existing Matrix terminals are allowed. The assembler must
support the contracted forms and operators needed to satisfy this requirement.

Among equivalent assemblable DAGs satisfying that requirement, compare costs
lexicographically: (rank-two assembly operations, all assembly operations).
Count shared nodes once. Coefficients, Matrix terminals and zeros cost nothing.
These costs do not estimate arithmetic operations or sparsity.

Tests exercise the public restructure_base_form interface and use explicit
equivalent forms to establish achievable costs. The cost check verifies
assemblability and counts operations without restructuring the input.
"""

import pytest
from utils import LagrangeElement

from ufl import (
    Action,
    Adjoint,
    Argument,
    Coargument,
    Coefficient,
    Cofunction,
    Form,
    FormSum,
    FunctionSpace,
    Matrix,
    Mesh,
    TestFunction,
    TrialFunction,
    action,
    adjoint,
    conj,
    derivative,
    dx,
    grad,
    inner,
    rhs,
    triangle,
)
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.restructure_base_form import restructure_base_form
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.form import BaseForm, ZeroBaseForm


def _assembly_cost(root):
    """Check structural assemblability, returning the cost of the unique DAG."""
    visited = set()
    operations = []

    def rank(node):
        return 1 if isinstance(node, Coefficient) else len(node.arguments())

    def symbolic(node):
        # These arguments are bound inside the containing form or operator.
        if isinstance(node, (Argument, Coargument)):
            return
        if isinstance(node, BaseForm):
            assert rank(node) <= 1, "Expected a rank-zero or rank-one dependency"
            visit(node)
        else:
            for operand in node.ufl_operands:
                symbolic(operand)

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        node_rank = rank(node)
        assert node_rank <= 2, f"Cannot assemble rank-{node_rank} {type(node).__name__}"
        if isinstance(node, (Coefficient, Cofunction, Matrix, ZeroBaseForm)):
            return
        if isinstance(node, BaseFormOperator):
            for value in (*node.ufl_operands, *node.argument_slots()):
                symbolic(value)
        elif isinstance(node, Form):
            for integral in node.integrals():
                symbolic(integral.integrand())
        elif isinstance(node, Action):
            for operand in node.ufl_operands:
                assert 1 <= rank(operand) <= 2, "Action operands must have rank one or two"
                visit(operand)
        elif isinstance(node, Adjoint):
            assert rank(node.form()) == 2, "Adjoint operand must have rank two"
            visit(node.form())
        elif isinstance(node, FormSum):
            for operand in node.components():
                assert rank(operand) == node_rank, "FormSum requires matching ranks"
                visit(operand)
        else:
            raise AssertionError(f"No assembly rule for {type(node).__name__}")
        operations.append(node_rank)

    visit(root)
    return operations.count(2), len(operations)


def _optimal_assembly_cost(root):
    """Check the rank-one optimality requirement, then return the DAG cost."""
    cost = _assembly_cost(root)
    if len(root.arguments()) == 1:
        assert cost[0] == 0, "A rank-one result must not assemble rank-two nodes"
    return cost


@pytest.fixture
def domain():
    return Mesh(LagrangeElement(triangle, 1, (2,)))


@pytest.fixture
def V1(domain):
    return FunctionSpace(domain, LagrangeElement(triangle, 1))


@pytest.fixture
def V2(domain):
    return FunctionSpace(domain, LagrangeElement(triangle, 2))


@pytest.fixture
def V3(domain):
    return FunctionSpace(domain, LagrangeElement(triangle, 3))


@pytest.fixture
def V4(domain):
    return FunctionSpace(domain, LagrangeElement(triangle, 4))


# --- Cost model ---


def test_assembly_cost_shared_operations(V1):
    A = Matrix(V1.dual(), V1)
    u = Coefficient(V1)
    shared = Action(A, u)
    result = FormSum((shared, 1), (Action(A, shared), 1))

    assert _assembly_cost(A) == (0, 0)
    assert _assembly_cost(result) == (0, 3)


def test_assembly_cost_operator_dependencies(V1):
    u = Coefficient(V1)
    vstar = Argument(V1.dual(), 0)
    direction = Interpolate(u, V1)
    applied = ExternalOperator(
        u, function_space=V1, derivatives=(1,), argument_slots=(vstar, direction)
    )
    assert _assembly_cost(applied) == (0, 2)
    assert _assembly_cost(direction * Argument(V1, 0) * dx) == (0, 2)

    B = Interpolate(Argument(V1, 1), V1)
    unsplit = ExternalOperator(u, function_space=V1, derivatives=(1,), argument_slots=(vstar, B))
    with pytest.raises(AssertionError, match="Expected a rank-zero or rank-one dependency"):
        _assembly_cost(unsplit)

    trilinear = Argument(V1, 0) * Argument(V1, 1) * Argument(V1, 2) * dx
    with pytest.raises(AssertionError, match="Cannot assemble rank-3 Form"):
        _assembly_cost(trilinear)


def test_rank_one_optimality(V1):
    u = Coefficient(V1)
    A = Matrix(V1.dual(), V1)
    bilinear = Argument(V1, 0) * Argument(V1, 1) * dx

    assert _optimal_assembly_cost(Action(A, u)) == (0, 1)
    assert _optimal_assembly_cost(bilinear) == (1, 1)
    # This DAG is assemblable, but fails the stronger rank-one requirement.
    expression = Action(bilinear, u)
    assert _assembly_cost(expression) == (1, 2)
    with pytest.raises(AssertionError, match="must not assemble rank-two nodes"):
        _optimal_assembly_cost(expression)


# --- Derivatives and preprocessing ---


@pytest.mark.parametrize("arity", [0, 1, 2])
def test_restructure_expands_interpolation_derivatives(V1, arity):
    u = Coefficient(V1)
    direction = Coefficient(V1)
    target = Cofunction(V1.dual()) if arity == 0 else V1
    if arity == 2:
        direction = Argument(V1, 1)
    expression = derivative(Interpolate(u, target), u, direction)

    actual = restructure_base_form(expression)

    assert actual == Interpolate(direction, target)
    assert len(actual.arguments()) == arity


@pytest.mark.parametrize("order", [1, 2])
def test_restructure_derivative_through_interpolation(V1, order):
    u = Coefficient(V1)
    expression = inner(Interpolate(u, V1), Interpolate(u, V1)) * dx
    for _ in range(order):
        expression = derivative(expression, u)
    expected = restructure_base_form(expand_derivatives(expression))

    actual = restructure_base_form(expression)

    assert actual == expected
    assert _optimal_assembly_cost(actual) == _optimal_assembly_cost(expected)


def test_restructure_ordinary_form(V1):
    u = Coefficient(V1)
    direction = Argument(V1, 0)
    form = derivative(inner(grad(u**2), grad(direction)) * dx, u)

    assert restructure_base_form(form) is form


@pytest.mark.parametrize("complex_mode", [False, True])
def test_restructure_preprocesses_forms_with_operators(V1, complex_mode):
    u = Coefficient(V1)
    v = Argument(V1, 0)
    operator = Interpolate(u, V1)
    form = inner(operator, v) * dx
    expected = (conj(v) if complex_mode else v) * operator * dx

    assert restructure_base_form(form, complex_mode=complex_mode) == expected


def test_restructure_preserves_compound_gradient(V1):
    u = Coefficient(V1)
    h = Coefficient(V1)
    expression = derivative(Interpolate(grad(u**2)[0], V1), u, h)

    assert restructure_base_form(expression) == Interpolate(grad(2 * h * u)[0], V1)


# --- Actions and Adjoints ---


def test_restructure_action_of_adjoint_matrix(V1, V2):
    # Action(Adjoint(A), x) -> Action(x, A)
    A = Matrix(V1.dual(), V2)
    f = Argument(V1, 0) * dx
    assert restructure_base_form(Action(Adjoint(A), f)) == Action(f, A)
    x = Coefficient(V1.dual())
    assert restructure_base_form(Action(Adjoint(A), x)) == Action(x, A)
    # The right operand has rank two, so the Adjoint stays.
    matrix_action = restructure_base_form(Action(Adjoint(A), Matrix(V1, V2)))
    assert isinstance(matrix_action.left(), Adjoint)


def test_restructure_action_of_adjoint_of_action(V1, V2):
    A = Matrix(V2.dual(), V1.dual())
    B = Matrix(V1, V2)
    x = Coefficient(V2.dual())

    # Action(Adjoint(Action(A, B)), x) -> Action(x, Action(A, B)) -> Action(Action(x, A), B)
    assert restructure_base_form(Action(Adjoint(Action(A, B)), x)) == Action(Action(x, A), B)


def test_restructure_action_of_adjoint_form(V1, V2):
    form = TestFunction(V1) * TrialFunction(V2) * dx
    x = Coefficient(V1)

    assert restructure_base_form(Action(Adjoint(form), x)) == conj(x * TestFunction(V2)) * dx


def test_restructure_adjoint_form_matrix_action(V1, V2):
    form = TestFunction(V2) * TrialFunction(V2) * dx
    B = Matrix(V2.dual(), V1)
    x = Coefficient(V2)

    restructured = restructure_base_form(Action(Adjoint(Action(form, B)), x))

    # The Form in the chain is assembled, so its argument is numbered from zero.
    assert restructured == Action(conj(x * TestFunction(V2)) * dx, B)


def test_restructure_adjoint_formsum_weights(V1, V2):
    A, B = Matrix(V2.dual(), V1), Matrix(V2.dual(), V1)
    x = Coefficient(V2.dual())

    restructured = restructure_base_form(Action(Adjoint(FormSum((A, 1 + 2j), (B, 3))), x))

    assert restructured == FormSum((Action(x, A), 1 - 2j), (Action(x, B), 3))


@pytest.mark.parametrize("matrix_first", [False, True], ids=["form-first", "matrix-first"])
def test_formsum_action_numbering(V1, V2, matrix_first):
    # Action(x, A) numbers its argument 1, while the Form that absorbs x numbers it 0.
    x = Coefficient(V2)
    A = Action(x, Matrix(V2, V1))
    F = restructure_base_form(Action(x, TestFunction(V2) * TrialFunction(V1) * dx))
    assert [a.number() for a in A.arguments()] == [1]
    assert [a.number() for a in F.arguments()] == [0]

    components = ((A, 1), (F, 1)) if matrix_first else ((F, 1), (A, 1))
    assert FormSum(*components).arguments() == F.arguments()


def test_formsum_empty_form(V1):
    v, u = TestFunction(V1), TrialFunction(V1)
    assert (rhs(v * u * dx) + Cofunction(V1.dual())).arguments() == (v,)


@pytest.mark.parametrize("sum_on_left", [False, True], ids=["right-sum", "left-sum"])
def test_restructure_distributes_expression_sum(V1, sum_on_left):
    a, b = Coefficient(V1), Coefficient(V1)
    if sum_on_left:
        A = Matrix(V1, V1)
        expression = Action(a + b, A)
        expected = (Action(a, A), Action(b, A))
    else:
        A = Matrix(V1.dual(), V1)
        expression = Action(A, a + b)
        expected = (Action(A, a), Action(A, b))

    assert restructure_base_form(expression) == FormSum(*((c, 1) for c in expected))


def test_restructure_form_absorbs_expression_sum(V1):
    # Absorption comes before distribution, so a Form absorbs a Sum whole.
    a, b = Coefficient(V1), Coefficient(V1)
    v, u = TestFunction(V1), TrialFunction(V1)

    assert restructure_base_form(Action(v * u * dx, a + b)) == v * (a + b) * dx


def test_restructure_adjoint_on_right(V1, V2):
    # Action(x, Adjoint(A)) -> Action(A, x)
    A = Matrix(V2.dual(), V1)
    u = Coefficient(V1)
    assert restructure_base_form(Action(u, Adjoint(A))) == Action(A, u)

    form = TrialFunction(V1) * TestFunction(V2) * dx
    assert restructure_base_form(Action(u, Adjoint(form))) == u * TestFunction(V2) * dx


def test_restructure_adjoint_operator_on_right(V1, V2):
    Iv = Interpolate(TrialFunction(V1), Argument(V2.dual(), 0))
    u = Coefficient(V1)

    # Action(u, Adjoint(Iv)) -> Action(u, I(v0; v1*)) -> I(u; v0*)
    actual = restructure_base_form(Action(u, Adjoint(Iv)))
    assert actual == Interpolate(u, Argument(V2.dual(), 0))
    assert _optimal_assembly_cost(actual) == (0, 1)


def test_restructure_adjoint_form_matrix_sum(V1, V2):
    x = Coefficient(V2)
    form = TestFunction(V2) * TrialFunction(V1) * dx
    expression = Action(Adjoint(FormSum((form, 1), (Matrix(V2, V1), 2))), x)

    actual = restructure_base_form(expression)
    assert isinstance(actual, FormSum)
    assert len(actual.arguments()) == 1
    assert _optimal_assembly_cost(actual) == (0, 3)


@pytest.mark.parametrize("terms", [("mass",), ("interpolation",), ("mass", "interpolation")])
def test_restructure_adjoint_action_arguments(V1, V2, terms):
    u = Coefficient(V1)
    v = TestFunction(V2)
    integrands = {"mass": inner(u, v) * dx, "interpolation": inner(Interpolate(u, V2), v) * dx}
    F = sum(integrands[term] for term in terms)
    expression = action(adjoint(derivative(F, u)), Coefficient(V2))

    restructured = restructure_base_form(expression)

    assert restructured.arguments() == expression.arguments() == (TestFunction(V1),)
    if isinstance(restructured, FormSum):
        assert all(c.arguments() == (TestFunction(V1),) for c in restructured.components())


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_matrix_subclass(V1, dual):
    class AssembledMatrix(Matrix):
        """An assembled operator supplied by another package."""

    matrix = AssembledMatrix(V1.dual(), V1)
    coefficient = Cofunction(V1.dual()) if dual else Coefficient(V1)
    operands = (coefficient, matrix) if dual else (matrix, coefficient)

    actual = restructure_base_form(Action(*operands))
    assert actual.left() is operands[0]
    assert actual.right() is operands[1]
    assert _optimal_assembly_cost(actual) == (0, 1)


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_rank_one_chain(V1, dual):
    A = Matrix(V1.dual(), V1)
    B = Matrix(V1.dual(), V1)
    if dual:
        expression = Action(Cofunction(V1.dual()), Action(A, B))
    else:
        expression = Action(Action(A, B), Coefficient(V1))

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (0, 2)
    assert restructure_base_form(actual) == actual


@pytest.mark.parametrize("sum_on_left", [False, True], ids=["right-sum", "left-sum"])
def test_restructure_keeps_matrix_sums_factored(V1, sum_on_left):
    A = Matrix(V1.dual(), V1)
    B = Matrix(V1.dual(), V1)
    C = Matrix(V1.dual(), V1)
    expression = Action(B + C, A) if sum_on_left else Action(A, B + C)
    # The input is already assemblable: one matrix sum and one matrix product.
    # Expanding it needs two matrix products and an additional matrix sum.
    assert _assembly_cost(expression) == (2, 2)

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (2, 2)


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_rank_one_formsum_chain(V1, dual):
    A = Matrix(V1.dual(), V1)
    B = Matrix(V1.dual(), V1)
    C = Matrix(V1.dual(), V1)
    if dual:
        x = Cofunction(V1.dual())
        expression = Action(Action(x, B + C), A)
        expected = Action(Action(x, B) + Action(x, C), A)
    else:
        x = Coefficient(V1)
        expression = Action(A, Action(B + C, x))
        expected = Action(A, Action(B, x) + Action(C, x))
    # Three rank-one Actions and one rank-one FormSum suffice, with the
    # rank-one operand on either side of the chain.
    assert expected.arguments() == expression.arguments()
    assert _assembly_cost(expected) == (0, 4)

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (0, 4)


def test_restructure_conjugates_matrix_sum_weights(V1):
    A = Matrix(V1.dual(), V1)
    B = Matrix(V1.dual(), V1)
    x = Cofunction(V1.dual())
    expression = Action(x, FormSum((A, 1j), (B, 1)))
    # (i A + B)^H x = -i A^H x + B^H x.
    expected = FormSum((Action(x, A), -1j), (Action(x, B), 1))

    actual = restructure_base_form(expression)
    assert actual == expected
    assert _optimal_assembly_cost(actual) == (0, 3)


@pytest.mark.parametrize("contractions", [1, 2])
def test_restructure_trilinear_form(V1, contractions):
    v = Argument(V1, 0)
    w = Argument(V1, 1)
    z = Argument(V1, 2)
    u = Coefficient(V1)
    direction = Coefficient(V1)
    expression = Action(v * w * z * dx, u)
    expected = v * w * u * dx
    if contractions == 2:
        expression = Action(expression, direction)
        expected = v * direction * u * dx
    cost = (int(contractions == 1), 1)
    # Substitute coefficients before assembly to avoid a rank-three Form.
    assert expected.arguments() == expression.arguments()
    assert _assembly_cost(expected) == cost

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == cost


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_bilinear_form_complex_direction(V1, dual):
    v = Argument(V1, 0)
    w = Argument(V1, 1)
    form = inner(w, v) * dx
    direction = Interpolate((1 + 2j) * Coefficient(V1), V1)
    expression = Action(direction, form) if dual else Action(form, direction)
    expected = conj(inner(v, direction)) * dx if dual else inner(direction, v) * dx

    actual = restructure_base_form(expression, complex_mode=True)
    assert actual == expected
    assert actual.arguments() == (v,)
    assert _optimal_assembly_cost(actual) == (0, 2)
    assert restructure_base_form(actual, complex_mode=True) == actual


# --- Operator argument slots ---


def test_restructure_fills_interpolate_operand(V1, V2):
    vstar = Argument(V2.dual(), 0)
    u = Coefficient(V1)
    Iu = Interpolate(u, vstar)
    Iv = Interpolate(TrialFunction(V1), vstar)  # V1 -> V2
    F = TrialFunction(V2) * TestFunction(V1) * dx

    # Interpolate is linear in its operand: Action(I(v1; v*), u) -> I(u; v*)
    assert restructure_base_form(Action(Iv, u)) == Iu
    # Associate before absorption, then assemble the resulting one-form directly.
    assert restructure_base_form(Action(Action(F, Iv), u)) == Iu * TestFunction(V1) * dx


def test_restructure_adjoint_operator_numbers(V1, V2):
    Iv = Interpolate(TrialFunction(V1), Argument(V2.dual(), 0))

    assert restructure_base_form(Adjoint(Iv)) == Interpolate(
        Argument(V1, 0), Argument(V2.dual(), 1)
    )


def test_restructure_adjoint_operator_dual_slot(V1, V2):
    Iv = Interpolate(TrialFunction(V1), Argument(V2.dual(), 0))
    w = Cofunction(V2.dual())

    # Operands are restructured before their Action:
    # Action(Adjoint(Iv), w) -> Action(I(v0; v1*), w) -> I(v0; w)
    assert restructure_base_form(Action(Adjoint(Iv), w)) == Interpolate(Argument(V1, 0), w)


def test_restructure_dual_slot_on_the_right(V1):
    u = Coefficient(V1)
    N = ExternalOperator(u, function_space=V1)
    vstar = Coargument(V1.dual(), 0)
    dNdu = N._ufl_expr_reconstruct_(u, derivatives=(1,), argument_slots=(vstar, Argument(V1, 1)))
    L = TestFunction(V1) * dx

    # Action(L, dN/du(u; v0*, v1)) -> dN/du(u; L, v0), since N is linear in its dual slot
    expected = N._ufl_expr_reconstruct_(u, derivatives=(1,), argument_slots=(L, Argument(V1, 0)))
    actual = Action(L, dNdu)
    actual = restructure_base_form(actual)
    assert actual == expected


def test_restructure_formsum_fills_dual_slot_whole(V1, V2):
    u = Coefficient(V1)
    L = TestFunction(V2) * dx + Cofunction(V2.dual())

    # A FormSum fills the dual slot without distributing the Action over it.
    assert restructure_base_form(Action(Interpolate(u, V2), L)) == Interpolate(u, L)


def test_restructure_splits_rank_2_dual_slot(V1, V2):
    u = Coefficient(V1)
    A = Matrix(V2, V1)

    # I(u; A) -> Action(A, I(u; v*))
    assert restructure_base_form(Interpolate(u, A)) == Action(
        A, Interpolate(u, Argument(V1.dual(), 0))
    )


def test_restructure_interp_hessian(V1, V2):
    # The residual is I(v0; dJ/dIu), with the dual slot depending on u.
    u = Coefficient(V1)
    f = Coefficient(V2)
    Iu = Interpolate(u, V2)
    v0, v1 = TestFunction(V1), TrialFunction(V1)
    w0, w1 = TestFunction(V2), TrialFunction(V2)
    J = 0.5 * (Iu - f) ** 2 * dx
    residual = restructure_base_form(derivative(J, u))
    assert isinstance(residual, Interpolate)

    hessian = restructure_base_form(derivative(residual, u, v1))

    # d2J/du2[v0, v1] = I^T d2J/dIu2 I
    d2JdIu2 = expand_derivatives(derivative(derivative(J, Iu, w0), Iu, w1))
    vhat = Argument(V2.dual(), 1)
    expected = Action(Interpolate(v0, vhat), Action(d2JdIu2, Interpolate(v1, V2)))
    assert hessian == expected
    assert hessian == restructure_base_form(derivative(derivative(J, u), u))
    _assembly_cost(hessian)


def test_restructure_external_operator_form_without_argument_0(V1, V2, V3, V4):
    w = Argument(V1, 0)
    free = Argument(V2, 1)
    dual_slot = inner(Argument(V3, 2), free) * dx
    N = ExternalOperator(Coefficient(V4), function_space=V3, argument_slots=(dual_slot, w))

    assert restructure_base_form(N).arguments() == (w, free)


def test_restructure_interpolate_form_without_argument_0(V1, V2, V3):
    w = Argument(V1, 0)
    free = Argument(V2, 1)
    dual_slot = inner(Argument(V3, 2), free) * dx

    restructured = restructure_base_form(Interpolate(w, dual_slot))

    assert restructured.arguments() == (w, free)


def test_restructure_keeps_operand_arguments(V1, V2):
    # The Coargument of an operand Interpolate is bound inside that operand.
    # Restructuring renumbers or contracts only the arguments in the argument
    # slots of the outer operator, so the operand stays the same.
    operand = Interpolate(Coefficient(V1), V2)
    vstar = Argument(V2.dual(), 0)
    uhat = Argument(V2, 1)
    N = ExternalOperator(operand, function_space=V2, argument_slots=(vstar, uhat))

    swapped = restructure_base_form(Adjoint(N))
    assert swapped.ufl_operands == (operand,)
    assert swapped.argument_slots() == (Argument(V2.dual(), 1), Argument(V2, 0))

    w = Cofunction(V2.dual())
    contracted = restructure_base_form(Action(w, N))
    assert contracted.ufl_operands == (operand,)
    assert contracted.argument_slots() == (w, Argument(V2, 0))

    u = Coefficient(V2)
    applied = restructure_base_form(Action(N, u))
    assert applied.ufl_operands == (operand,)
    assert applied.argument_slots() == (vstar, u)


def test_restructure_computed_primal_direction(V1):
    u = Coefficient(V1)
    vstar = Argument(V1.dual(), 0)
    direction = Interpolate(u, V1)
    operator = ExternalOperator(
        u, function_space=V1, derivatives=(1,), argument_slots=(vstar, Argument(V1, 1))
    )
    expression = Action(operator, direction)
    expected = ExternalOperator(
        u, function_space=V1, derivatives=(1,), argument_slots=(vstar, direction)
    )
    # One interpolation and one action of the derivative; both have rank one.
    assert _assembly_cost(expected) == (0, 2)
    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (0, 2)


# --- Operator compositions ---


def test_restructure_splits_rank_2_operand_slot(V1, V2):
    # The chain rule puts the rank 2 derivative B of the operand into an
    # argument slot. A base form operator is linear in that slot, so the
    # restructurer splits it off: N(u; v*, B) -> Action(N(u; v*, uhat), B).
    u = Coefficient(V1)
    operand = Interpolate(u, V2)
    N = ExternalOperator(operand, function_space=V2)
    vstar = Argument(V2.dual(), 0)
    B = Interpolate(TrialFunction(V1), V2)

    dN = expand_derivatives(derivative(N, u))
    assert dN == N._ufl_expr_reconstruct_(operand, derivatives=(1,), argument_slots=(vstar, B))

    left = N._ufl_expr_reconstruct_(
        operand, derivatives=(1,), argument_slots=(vstar, Argument(V2, 1))
    )
    assert restructure_base_form(dN) == Action(left, B)


def test_restructure_composition_in_interpolate_slot(V1):
    # Interpolating A o A is the same three-map composition however it is associated.
    A = Interpolate(TrialFunction(V1), V1)
    ustar = Cofunction(V1.dual())

    restructured = restructure_base_form(Action(ustar, Interpolate(Action(A, A), V1)))

    assert restructured == restructure_base_form(Action(ustar, Action(A, Action(A, A))))
    assert restructured.arguments() == (TestFunction(V1),)
    assert restructure_base_form(restructured) == restructured


def test_restructure_weighted_sum_in_interpolate_slot(V1):
    # Interpolating 2 A + 3 B is 2 A o A + 3 A o B, with A the interpolation of a trial function.
    A = Interpolate(TrialFunction(V1), V1)
    B = Interpolate(2 * TrialFunction(V1), V1)
    ustar = Cofunction(V1.dual())

    restructured = restructure_base_form(Action(ustar, Interpolate(FormSum((A, 2), (B, 3)), V1)))

    assert restructured == FormSum(
        (restructure_base_form(Action(ustar, Action(A, A))), 2),
        (restructure_base_form(Action(ustar, Action(A, B))), 3),
    )
    assert restructured.arguments() == (TestFunction(V1),)
    assert restructure_base_form(restructured) == restructured


def test_restructure_contracts_covector_first(V1, V2):
    # A rank 1 operand on the left of a chain of Actions is contracted first:
    # Action(x, Action(A, dN)) contracts x into A before filling dN's dual
    # slot, so neither operand needs a rank-two assembly.
    u = Coefficient(V1)
    N = ExternalOperator(u, function_space=V2)
    dN = N._ufl_expr_reconstruct_(
        u, derivatives=(1,), argument_slots=(Argument(V2.dual(), 0), Argument(V1, 1))
    )
    A = inner(TrialFunction(V2), TestFunction(V1)) * dx
    x = Coefficient(V1)

    expected = N._ufl_expr_reconstruct_(
        u,
        derivatives=(1,),
        argument_slots=(conj(inner(Argument(V2, 0), x)) * dx, Argument(V1, 0)),
    )
    assert restructure_base_form(Action(x, Action(A, dN))) == expected
    assert restructure_base_form(Action(Adjoint(Action(A, dN)), x)) == expected


@pytest.mark.parametrize("adjoint", [False, True], ids=["jacobian", "adjoint"])
def test_restructure_action_of_composition(V1, adjoint):
    # Jacobian of F = inner(N(I(u)), v) * dx: dF/dN, dN/dI, dI/du.
    w = Coefficient(V1)
    vstar = Argument(V1.dual(), 0)
    v = Argument(V1, 0)
    operand = Interpolate(Coefficient(V1), V1)
    dFdN = inner(Argument(V1, 1), v) * dx
    dNdI = ExternalOperator(
        operand, function_space=V1, derivatives=(1,), argument_slots=(vstar, Argument(V1, 1))
    )
    dIdu = Interpolate(Argument(V1, 1), V1)
    jacobian = Action(dFdN, Action(dNdI, dIdu))
    if adjoint:
        expression = Action(Adjoint(jacobian), w)
        # dI/du*[dN/dI*(dF/dN*[w], v)]
        slots = (conj(inner(v, w)) * dx, v)
        expected = Interpolate(
            v, dNdI._ufl_expr_reconstruct_(operand, derivatives=(1,), argument_slots=slots)
        )
    else:
        expression = Action(jacobian, w)
        # dF/dN[dN/dI[dI/du[w]]]
        slots = (vstar, Interpolate(w, V1))
        applied = dNdI._ufl_expr_reconstruct_(operand, derivatives=(1,), argument_slots=slots)
        expected = inner(applied, v) * dx

    actual = restructure_base_form(expression, complex_mode=True)
    assert actual == expected
    assert actual.arguments() == expression.arguments()
    # One Form and three operators, each assembled with rank one.
    assert _optimal_assembly_cost(actual) == (0, 4)
    assert restructure_base_form(actual, complex_mode=True) == actual


def test_restructure_hessian_action_composition(V1):
    u = Coefficient(V1)
    vstar = Argument(V1.dual(), 0)
    first = Interpolate(Argument(V1, 1), V1)
    second = Interpolate(Argument(V1, 2), V1)
    hessian = ExternalOperator(
        u, function_space=V1, derivatives=(2,), argument_slots=(vstar, first, second)
    )
    expression = Action(hessian, u)
    # Contract the second direction, then expose the remaining composition.
    contracted = ExternalOperator(
        u,
        function_space=V1,
        derivatives=(2,),
        argument_slots=(vstar, Argument(V1, 1), Interpolate(u, V1)),
    )
    expected = Action(contracted, first)
    assert expected.arguments() == expression.arguments()
    assert _assembly_cost(expected) == (3, 4)

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    # Rank two at the root is insufficient: its remaining rank-two slot
    # cannot be assembled into a coefficient for the external operator.
    assert _optimal_assembly_cost(actual) == (3, 4)
    assert restructure_base_form(actual) == actual


def test_restructure_hessian_two_actions(V1):
    u = Coefficient(V1)
    direction = Coefficient(V1)
    vstar = Argument(V1.dual(), 0)
    first = Interpolate(Argument(V1, 1), V1)
    second = Interpolate(Argument(V1, 2), V1)
    hessian = ExternalOperator(
        u, function_space=V1, derivatives=(2,), argument_slots=(vstar, first, second)
    )
    expression = Action(Action(hessian, u), direction)
    expected = ExternalOperator(
        u,
        function_space=V1,
        derivatives=(2,),
        argument_slots=(vstar, Interpolate(direction, V1), Interpolate(u, V1)),
    )

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    # Completing the inner contraction exposes an Action in the parent's
    # left operand. Reassociate that parent before trying its contraction.
    assert _optimal_assembly_cost(actual) == (0, 3)
    assert actual == expected
    assert restructure_base_form(actual) == actual
