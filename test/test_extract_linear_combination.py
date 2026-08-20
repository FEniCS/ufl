import pytest
from utils import LagrangeElement

import ufl
from ufl.algorithms.extract_linear_combination import extract_linear_combination


@pytest.fixture
def domain():
    return ufl.Mesh(LagrangeElement(ufl.triangle, 1, (2,)))


@pytest.fixture
def V(domain):
    """Standard spatial function space."""
    el = LagrangeElement(domain.ufl_cell(), 3)
    return ufl.FunctionSpace(domain, el)


@pytest.fixture
def V_vec(domain):
    """Vector function space for indexed tests."""
    return ufl.FunctionSpace(domain, LagrangeElement(domain.ufl_cell(), 2, (3,)))


def test_valid_linear_combinations(V):
    """Test standard valid combinations.

    Sum, Product, Division, Negative, and float/int terminals.
    """
    u = ufl.Coefficient(V)
    v = ufl.Coefficient(V)

    # Includes: IntValue (3), FloatValue (1.5), Sum, Negative (-v/4.0), Product, Division
    expr = 2.0 * u - v / 4.0 + ufl.as_ufl(3) * u - ufl.classes.FloatValue(1.5) * v

    res = extract_linear_combination(expr)

    assert len(res) == 4
    # The DAG traversal evaluates leaves to roots, maintaining term extraction
    # Order might depend slightly on UFL's internal DAG sorting, but usually
    # follows algebraic definition
    assert (2.0, u) in res
    assert (-0.25, v) in res
    assert (3.0, u) in res
    assert (-1.5, v) in res


def test_scalars_and_powers(V, domain):
    """Test evaluating constants, real functions, and powers as scalar weights."""
    u = ufl.Coefficient(V)

    c = ufl.Constant(domain)
    d = ufl.Constant(domain)
    expr = (d**2) * u + (c + d**2) * u

    res = extract_linear_combination(expr)

    assert len(res) == 2
    assert (d**2, u) in res
    assert (c + d**2, u) in res


def test_ufl_zero_node(V):
    """Test that UFL Zero nodes are handled correctly (usually evaluates to an empty list)."""
    u = ufl.Coefficient(V)

    # Create an explicit Zero node with the shape of u
    z = ufl.classes.Zero(u.ufl_shape, u.ufl_free_indices, u.ufl_index_dimensions)

    expr = u + z
    res = extract_linear_combination(expr)

    assert len(res) == 1
    assert res[0] == (1.0, u)


def test_nonlinear_errors(V):
    """Test that non-linear operations strictly raise ValueErrors."""
    u = ufl.Coefficient(V)
    v = ufl.Coefficient(V)

    with pytest.raises(ValueError, match="product of two spatial functions"):
        extract_linear_combination(u * v)

    with pytest.raises(ValueError, match="division by a spatial function"):
        extract_linear_combination(u / v)

    with pytest.raises(ValueError, match="power involving a spatial function"):
        extract_linear_combination(u**2)


def test_invalid_additions(V, domain):
    """Test that adding a scalar to a spatial function is caught."""
    u = ufl.Coefficient(V)
    c = ufl.Constant(domain)

    with pytest.raises(
        ValueError, match="Cannot directly add a raw scalar expression to a spatial function"
    ):
        extract_linear_combination(u + c)

    with pytest.raises(
        ValueError, match="Cannot directly add a raw scalar expression to a spatial function"
    ):
        extract_linear_combination(u + 5.0)


def test_pure_scalar_error(domain):
    """Test that evaluating an expression with NO spatial functions raises an error."""
    c = ufl.Constant(domain)
    r_func = ufl.Constant(domain)

    expr = c * 5.0 + (r_func**2)

    with pytest.raises(ValueError, match="Expression evaluated to a pure scalar"):
        extract_linear_combination(expr)


def test_forbidden_operations(V_vec):
    """Test that indexed vectors and component tensors trigger a NotImplementedError."""
    u = ufl.Coefficient(V_vec)

    with pytest.raises(
        NotImplementedError, match="Direct array assignment of indexed vector components"
    ):
        extract_linear_combination(u[0])


def test_negative(V, domain):
    """Test that negative operations are handled correctly."""
    r_func = ufl.Constant(domain)
    u = ufl.Coefficient(V)
    expr = -r_func * u + 2 * u

    res = extract_linear_combination(expr)

    assert len(res) == 2

    assert (-r_func, u) in res
    assert (2.0, u) in res
