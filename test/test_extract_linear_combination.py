# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from utils import LagrangeElement, MixedElement

import ufl
from ufl.algorithms.extract_linear_combination import extract_linear_combination
from ufl.algorithms.renumbering import renumber_indices


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


def test_negative(V, domain):
    """Test that negative operations are handled correctly."""
    r_func = ufl.Constant(domain)
    u = ufl.Coefficient(V)
    expr = -r_func * u + 2 * u

    res = extract_linear_combination(expr)

    assert len(res) == 2

    assert (-r_func, u) in res
    assert (2.0, u) in res


def test_matrix_linear_combination(V, domain):
    """Test linear combinations involving ufl.Matrix."""
    # Matrix requires a row space and column space
    A = ufl.Matrix(V, V)
    c = ufl.Constant(domain)
    expr = 2.0 * A + (0.3 + c**2) * A

    res = extract_linear_combination(expr)

    assert len(res) == 2
    # NOTE: Matrices store scalar weights under `weights`
    assert (2.0, A) in res
    assert (0.3 + c**2, A) in res


def test_cofunction_linear_combination(V):
    """Test linear combinations involving ufl.Cofunction."""
    # Cofunction requires a dual space (it will raise an error if given a primal space)
    V_dual = V.dual()
    c = ufl.Cofunction(V_dual)
    d = ufl.Constant(V.ufl_domain())
    expr = -4 * d * c + 5.0 * c
    res = extract_linear_combination(expr)

    assert len(res) == 2
    assert (-4 * d, c) in res
    assert (5.0, c) in res


def test_matrix_nonlinear_error(V):
    """Test that matrices are protected by the same non-linear guardrails."""
    A = ufl.Matrix(V, V)
    u = ufl.Coefficient(V)

    with pytest.raises(
        ValueError, match=r"Non-linear expression detected: product of two spatial functions."
    ):
        # Cannot multiply a matrix by a coefficient algebraically in this block
        extract_linear_combination(A * u)


def test_vector_scalar_multiplication(V_vec, domain):
    """Test that UFL's ComponentTensor representation of vector-scalar multiplication
    is cleanly resolved back into the full operator."""
    u = ufl.Coefficient(V_vec)
    dt = ufl.Constant(domain)

    # Multiplying a vector Coefficient by a scalar Constant implicitly wraps
    # the expression in ComponentTensor(Indexed(...), ...).
    expr = u * dt
    res = extract_linear_combination(expr)

    assert len(res) == 1
    weight, func = res[0]

    # The traverser should have collapsed the implicit indexing back down
    # to the base coefficient and isolated the scalar weight.
    assert weight == dt
    assert func == u


def test_explicit_vector_indexing_with_weights_blocked(V_vec, domain):
    """Test that distributing a weight across an explicitly indexed vector field is blocked."""
    u = ufl.Coefficient(V_vec)
    dt = ufl.Constant(domain)

    expr = 5.0 * dt * u[1]

    pairs = extract_linear_combination(expr)
    assert len(pairs) == 1
    assert pairs[0] == (5.0 * dt, u[1])


def test_tensor_scalar_multiplication(domain):
    """Test that matrix-scalar multiplication safely collapses back to the whole operator."""
    # Create a rank-2 tensor space
    gdim = domain.geometric_dimension
    V_tensor = ufl.FunctionSpace(
        domain,
        LagrangeElement(domain.ufl_cell(), 2, (gdim, gdim)),
    )
    T = ufl.Coefficient(V_tensor)
    dt = ufl.Constant(domain)

    expr = T * dt

    res = extract_linear_combination(expr)

    assert len(res) == 1
    weight, func = res[0]

    assert weight == dt
    assert func == T


def test_explicit_tensor_indexing_blocked(domain):
    """Test that explicitly accessing components of a rank-2 tensor is blocked."""
    gdim = domain.geometric_dimension
    V_tensor = ufl.FunctionSpace(
        domain,
        LagrangeElement(domain.ufl_cell(), 2, (gdim, gdim)),
    )
    T = ufl.Coefficient(V_tensor)
    res = extract_linear_combination(T[0, 1])
    assert len(res) == 1
    assert res[0] == (1.0, T[0, 1])


def test_mixed_element_extraction(domain):
    """Test extracting a linear combination from a mixed element component."""
    # Define a mixed element
    e1 = LagrangeElement(domain.ufl_cell(), 1)
    e2 = LagrangeElement(domain.ufl_cell(), 2, shape=(2,))
    mixed_element = MixedElement([e1, e2])

    W = ufl.FunctionSpace(domain, mixed_element)
    w = ufl.Coefficient(W)

    a = ufl.Constant(domain)
    b = ufl.Constant(domain)
    c = ufl.Constant(domain)

    expr = a * w[0] + b * w[1] + c * b * w[2]

    res = extract_linear_combination(expr)

    assert len(res) == 3

    assert (a, w[0]) in res
    assert (b, w[1]) in res
    assert (c * b, w[2]) in res


def test_explicit_vector_indexing_supported(V_vec, domain):
    """Test extracting a linear combination from explicitly indexed vector components."""
    u = ufl.Coefficient(V_vec)
    a = ufl.Constant(domain)
    b = ufl.Constant(domain)

    # Extracting standard vector components
    expr = a * u[0] + b * u[1]

    res = extract_linear_combination(expr)

    assert len(res) == 2
    assert (a, u[0]) in res
    assert (b, u[1]) in res


def test_explicit_tensor_indexing_supported(domain):
    """Test extracting a linear combination from explicitly indexed tensor components."""
    gdim = domain.geometric_dimension
    V_tensor = ufl.FunctionSpace(
        domain,
        LagrangeElement(domain.ufl_cell(), 2, (gdim, gdim)),
    )
    T = ufl.Coefficient(V_tensor)
    dt = ufl.Constant(domain)

    expr = dt * T[0, :] + 5.0 * T[1, :]

    res = extract_linear_combination(expr)
    res = [(w, renumber_indices(f)) for w, f in res]
    assert len(res) == 2
    assert (dt, renumber_indices(T[0, :])) in res
    assert (5.0, renumber_indices(T[1, :])) in res
