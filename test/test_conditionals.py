__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-08-20 -- 2012-11-30"

import pytest

from ufl import Coefficient, conditional, eq, ge, gt, le, lt, ne, triangle
from ufl.classes import EQ, GE, GT, LE, LT, NE
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1


@pytest.fixture
def f():
    element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    return Coefficient(element)


@pytest.fixture
def g():
    element = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    return Coefficient(element)


def test_conditional_does_not_allow_bool_condition(f, g):
    # The reason for this test is that it protects from the case
    # conditional(a == b, t, f) in which a == b means comparing representations
    with pytest.raises(BaseException):
        conditional(True, 1, 0)


def test_eq_oper_produces_bool(f, g):
    expr1 = f == f
    expr2 = f == g
    assert isinstance(expr1, bool)
    assert isinstance(expr2, bool)
    assert expr1
    assert not expr2


def test_eq_produces_ufl_expr(f, g):
    expr1 = eq(f, g)
    expr2 = eq(f, f)
    expr3 = f == g
    expr4 = f == f
    # Correct types:
    assert isinstance(expr1, EQ)
    assert isinstance(expr2, EQ)
    assert isinstance(expr3, bool)
    assert isinstance(expr4, bool)
    # Comparing representations correctly:
    assert bool(expr1 == eq(f, g))
    assert bool(expr1 != eq(g, g))
    assert bool(expr2 == eq(f, f))
    assert bool(expr2 != eq(g, f))
    # Bool evaluation yields actual bools:
    assert isinstance(bool(expr1), bool)
    assert isinstance(bool(expr2), bool)
    assert not expr3
    assert expr4
    # Allow use in boolean python expression context:
    # NB! This means comparing representations! Required by dict and set.
    assert not bool(expr1)
    assert bool(expr2)
    assert not bool(expr3)
    assert bool(expr4)


def test_ne_produces_ufl_expr(f, g):
    expr1 = ne(f, g)
    expr2 = ne(f, f)
    expr3 = f != g
    expr4 = f != f
    # Correct types:
    assert isinstance(expr1, NE)
    assert isinstance(expr2, NE)
    assert isinstance(expr3, bool)
    assert isinstance(expr4, bool)
    # Comparing representations correctly:
    assert bool(expr1 == ne(f, g))
    assert bool(expr1 != ne(g, g))
    assert bool(expr2 == ne(f, f))
    assert bool(expr2 != ne(g, f))
    assert not bool(expr2 == expr3)
    # Bool evaluation yields actual bools:
    assert isinstance(bool(expr1), bool)
    assert isinstance(bool(expr2), bool)
    # Allow use in boolean python expression context:
    # NB! This means the opposite of ==, i.e. comparing representations!
    assert bool(expr1)
    assert not bool(expr2)
    assert bool(expr1)
    assert not bool(expr2)


def test_lt_produces_ufl_expr(f, g):
    expr1 = lt(f, g)
    expr2 = f < g
    # Correct types (no bools here!):
    assert isinstance(expr1, LT)
    assert isinstance(expr2, LT)
    # Representations are the same:
    assert bool(expr1 == expr2)
    # Protection from misuse in boolean python expression context:
    with pytest.raises(BaseException):
        bool(expr1)


def test_gt_produces_ufl_expr(f, g):
    expr1 = gt(f, g)
    expr2 = f > g
    # Correct types (no bools here!):
    assert isinstance(expr1, GT)
    assert isinstance(expr2, GT)
    # Representations are the same:
    assert bool(expr1 == expr2)
    # Protection from misuse in boolean python expression context:
    with pytest.raises(BaseException):
        bool(expr1)


def test_le_produces_ufl_expr(f, g):
    expr1 = le(f, g)
    expr2 = f <= g
    # Correct types (no bools here!):
    assert isinstance(expr1, LE)
    assert isinstance(expr2, LE)
    # Representations are the same:
    assert bool(expr1 == expr2)
    # Protection from misuse in boolean python expression context:
    with pytest.raises(BaseException):
        bool(expr1)


def test_ge_produces_ufl_expr(f, g):
    expr1 = ge(f, g)
    expr2 = f >= g
    # Correct types (no bools here!):
    assert isinstance(expr1, GE)
    assert isinstance(expr2, GE)
    # Representations are the same:
    assert bool(expr1 == expr2)
    # Protection from misuse in boolean python expression context:
    with pytest.raises(BaseException):
        bool(expr1)
