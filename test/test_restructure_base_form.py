"""Assemblability and cost specifications for BaseForm restructuring.

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
    conj,
    dx,
    inner,
    triangle,
)
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
def V():
    mesh = Mesh(LagrangeElement(triangle, 1, (2,)))
    return FunctionSpace(mesh, LagrangeElement(triangle, 1))


def test_assembly_cost_counts_shared_operations_once(V):
    A = Matrix(V.dual(), V)
    u = Coefficient(V)
    shared = Action(A, u)
    result = FormSum((shared, 1), (Action(A, shared), 1))

    assert _assembly_cost(A) == (0, 0)
    assert _assembly_cost(result) == (0, 3)


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_preserves_an_assembled_matrix_subclass(V, dual):
    class AssembledMatrix(Matrix):
        """An assembled operator supplied by another package."""

    matrix = AssembledMatrix(V.dual(), V)
    coefficient = Cofunction(V.dual()) if dual else Coefficient(V)
    operands = (coefficient, matrix) if dual else (matrix, coefficient)

    actual = restructure_base_form(Action(*operands))
    assert actual.left() is operands[0]
    assert actual.right() is operands[1]
    assert _optimal_assembly_cost(actual) == (0, 1)


def test_assembly_cost_checks_operator_dependencies(V):
    u = Coefficient(V)
    vstar = Argument(V.dual(), 0)
    direction = Interpolate(u, V)
    applied = ExternalOperator(
        u, function_space=V, derivatives=(1,), argument_slots=(vstar, direction)
    )
    assert _assembly_cost(applied) == (0, 2)
    assert _assembly_cost(direction * Argument(V, 0) * dx) == (0, 2)

    B = Interpolate(Argument(V, 1), V)
    unsplit = ExternalOperator(u, function_space=V, derivatives=(1,), argument_slots=(vstar, B))
    with pytest.raises(AssertionError, match="Expected a rank-zero or rank-one dependency"):
        _assembly_cost(unsplit)

    trilinear = Argument(V, 0) * Argument(V, 1) * Argument(V, 2) * dx
    with pytest.raises(AssertionError, match="Cannot assemble rank-3 Form"):
        _assembly_cost(trilinear)


def test_rank_one_optimality_distinguishes_existing_and_new_matrices(V):
    u = Coefficient(V)
    A = Matrix(V.dual(), V)
    bilinear = Argument(V, 0) * Argument(V, 1) * dx

    assert _optimal_assembly_cost(Action(A, u)) == (0, 1)
    assert _optimal_assembly_cost(bilinear) == (1, 1)
    # This DAG is assemblable, but fails the stronger rank-one requirement.
    expression = Action(bilinear, u)
    assert _assembly_cost(expression) == (1, 2)
    with pytest.raises(AssertionError, match="must not assemble rank-two nodes"):
        _optimal_assembly_cost(expression)


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_avoids_rank_two_actions_in_rank_one_chains(V, dual):
    A, B = (Matrix(V.dual(), V) for _ in range(2))
    if dual:
        expression = Action(Cofunction(V.dual()), Action(A, B))
    else:
        expression = Action(Action(A, B), Coefficient(V))

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (0, 2)
    assert restructure_base_form(actual) == actual


def test_restructure_applies_computed_primal_direction_without_a_matrix(V):
    u = Coefficient(V)
    vstar = Argument(V.dual(), 0)
    direction = Interpolate(u, V)
    operator = ExternalOperator(
        u, function_space=V, derivatives=(1,), argument_slots=(vstar, Argument(V, 1))
    )
    expression = Action(operator, direction)
    witness = ExternalOperator(
        u, function_space=V, derivatives=(1,), argument_slots=(vstar, direction)
    )
    # One interpolation and one action of the derivative; both have rank one.
    assert _assembly_cost(witness) == (0, 2)
    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (0, 2)


def test_restructure_exposes_operator_composition_after_contracting_a_hessian(V):
    u = Coefficient(V)
    vstar = Argument(V.dual(), 0)
    first = Interpolate(Argument(V, 1), V)
    second = Interpolate(Argument(V, 2), V)
    hessian = ExternalOperator(
        u, function_space=V, derivatives=(2,), argument_slots=(vstar, first, second)
    )
    expression = Action(hessian, u)
    # Contract the second direction, then expose the remaining composition.
    contracted = ExternalOperator(
        u,
        function_space=V,
        derivatives=(2,),
        argument_slots=(vstar, Argument(V, 1), Interpolate(u, V)),
    )
    witness = Action(contracted, first)
    assert witness.arguments() == expression.arguments()
    assert _assembly_cost(witness) == (3, 4)

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    # Rank two at the root is insufficient: its remaining rank-two slot
    # cannot be assembled into a coefficient for the external operator.
    assert _optimal_assembly_cost(actual) == (3, 4)
    assert restructure_base_form(actual) == actual


def test_restructure_contracts_a_newly_exposed_composition_from_its_parent(V):
    u, direction = Coefficient(V), Coefficient(V)
    vstar = Argument(V.dual(), 0)
    first = Interpolate(Argument(V, 1), V)
    second = Interpolate(Argument(V, 2), V)
    hessian = ExternalOperator(
        u, function_space=V, derivatives=(2,), argument_slots=(vstar, first, second)
    )
    expression = Action(Action(hessian, u), direction)
    witness = ExternalOperator(
        u,
        function_space=V,
        derivatives=(2,),
        argument_slots=(vstar, Interpolate(direction, V), Interpolate(u, V)),
    )

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    # Completing the inner contraction exposes an Action in the parent's
    # left operand. Reassociate that parent before trying its contraction.
    assert _optimal_assembly_cost(actual) == (0, 3)
    assert actual == witness
    assert restructure_base_form(actual) == actual


@pytest.mark.parametrize("sum_on_left", [False, True], ids=["right-sum", "left-sum"])
def test_restructure_keeps_matrix_sums_factored(V, sum_on_left):
    A, B, C = (Matrix(V.dual(), V) for _ in range(3))
    expression = Action(B + C, A) if sum_on_left else Action(A, B + C)
    # The input is already assemblable: one matrix sum and one matrix product.
    # Expanding it needs two matrix products and an additional matrix sum.
    assert _assembly_cost(expression) == (2, 2)

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (2, 2)


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_factors_a_shared_action_on_a_rank_one_formsum(V, dual):
    A, B, C = (Matrix(V.dual(), V) for _ in range(3))
    if dual:
        x = Cofunction(V.dual())
        expression = Action(Action(x, B + C), A)
        witness = Action(Action(x, B) + Action(x, C), A)
    else:
        x = Coefficient(V)
        expression = Action(A, Action(B + C, x))
        witness = Action(A, Action(B, x) + Action(C, x))
    # Three rank-one Actions and one rank-one FormSum suffice, with the
    # rank-one operand on either side of the chain.
    assert witness.arguments() == expression.arguments()
    assert _assembly_cost(witness) == (0, 4)

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (0, 4)


@pytest.mark.parametrize("contractions", [1, 2])
def test_restructure_contracts_a_trilinear_form_before_assembly(V, contractions):
    v, w, z = (Argument(V, i) for i in range(3))
    u, direction = Coefficient(V), Coefficient(V)
    expression = Action(v * w * z * dx, u)
    witness = v * w * u * dx
    if contractions == 2:
        expression = Action(expression, direction)
        witness = v * direction * u * dx
    cost = (int(contractions == 1), 1)
    # Substitute coefficients before assembly to avoid a rank-three Form.
    assert witness.arguments() == expression.arguments()
    assert _assembly_cost(witness) == cost

    actual = restructure_base_form(expression)
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == cost


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
def test_restructure_contracts_bilinear_forms_with_complex_directions(V, dual):
    v, w = Argument(V, 0), Argument(V, 1)
    form = inner(w, v) * dx
    direction = Interpolate((1 + 2j) * Coefficient(V), V)
    expression = Action(direction, form) if dual else Action(form, direction)
    witness = conj(inner(w, direction)) * dx if dual else inner(direction, v) * dx

    actual = restructure_base_form(expression)
    assert actual == witness
    assert actual.arguments() == expression.arguments()
    assert _optimal_assembly_cost(actual) == (0, 2)
    assert restructure_base_form(actual) == actual


def test_restructure_conjugates_weights_when_contracting_a_matrix_sum_from_the_left(V):
    A, B = (Matrix(V.dual(), V) for _ in range(2))
    x = Cofunction(V.dual())
    expression = Action(x, FormSum((A, 1j), (B, 1)))
    # (i A + B)^H x = -i A^H x + B^H x.
    witness = FormSum((Action(x, A), -1j), (Action(x, B), 1))

    actual = restructure_base_form(expression)
    assert actual == witness
    assert _optimal_assembly_cost(actual) == (0, 3)
