import math

import pytest
from numpy import ndindex, reshape

from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    TestFunction,
    TrialFunction,
    VectorConstant,
    acos,
    as_tensor,
    as_ufl,
    asin,
    atan,
    cos,
    cosh,
    dx,
    exp,
    i,
    j,
    ln,
    max_value,
    min_value,
    outer,
    sin,
    sinh,
    tan,
    tanh,
    triangle,
)
from ufl.algorithms import compute_form_data
from ufl.constantvalue import Zero
from ufl.core.multiindex import FixedIndex, Index, MultiIndex, indices
from ufl.finiteelement import FiniteElement
from ufl.indexed import Indexed
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1
from ufl.tensors import ComponentTensor, ListTensor


def xtest_zero_times_argument(self):
    # FIXME: Allow zero forms
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    L = 0 * v * dx
    a = 0 * (u * v) * dx
    b = (0 * u) * v * dx
    assert len(compute_form_data(L).arguments) == 1
    assert len(compute_form_data(a).arguments) == 2
    assert len(compute_form_data(b).arguments) == 2


def test_divisions(self):
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)

    # Test simplification of division by 1
    a = f
    b = f / 1
    assert a == b

    # Test simplification of division by 1.0
    a = f
    b = f / 1.0
    assert a == b

    # Test simplification of division by of zero by something
    a = 0 / f
    b = 0 * f
    assert a == b

    # Test simplification of division by self (this simplification has been disabled)
    # a = f/f
    # b = 1
    # assert a == b


def test_products(self):
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    g = Coefficient(space)

    # Test simplification of literal multiplication
    assert f * 0 == as_ufl(0)
    assert 0 * f == as_ufl(0)
    assert 1 * f == f
    assert f * 1 == f
    assert as_ufl(2) * as_ufl(3) == as_ufl(6)
    assert as_ufl(2.0) * as_ufl(3.0) == as_ufl(6.0)

    # Test reordering of operands
    assert f * g == g * f

    # Test simplification of self-multiplication (this simplification has been disabled)
    # assert f*f == f**2


def test_sums(self):
    element = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    g = Coefficient(space)

    # Test reordering of operands
    assert f + g == g + f

    # Test adding zero
    assert f + 0 == f
    assert 0 + f == f

    # Test collapsing of basic sum (this simplification has been disabled)
    # assert f + f == 2 * f

    # Test reordering of operands and collapsing sum
    a = f + g + f  # not collapsed, but ordered
    b = g + f + f  # not collapsed, but ordered
    c = (g + f) + f  # not collapsed, but ordered
    d = f + (f + g)  # not collapsed, but ordered
    assert a == b
    assert a == c
    assert a == d

    # Test reordering of operands and collapsing sum
    a = f + f + g  # collapsed
    b = g + (f + f)  # collapsed
    assert a == b


def test_mathfunctions(self):
    for a in (0.1, 0.3, 0.9):
        assert math.sin(a) == sin(a)
        assert math.cos(a) == cos(a)
        assert math.tan(a) == tan(a)
        assert math.sinh(a) == sinh(a)
        assert math.cosh(a) == cosh(a)
        assert math.tanh(a) == tanh(a)
        assert math.asin(a) == asin(a)
        assert math.acos(a) == acos(a)
        assert math.atan(a) == atan(a)
        assert math.exp(a) == exp(a)
        assert math.log(a) == ln(a)
        # TODO: Implement automatic simplification of conditionals?
        assert a == float(max_value(a, a - 1))
        # TODO: Implement automatic simplification of conditionals?
        assert a == float(min_value(a, a + 1))


def test_indexing(self):
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    u = VectorConstant(domain)
    v = VectorConstant(domain)

    A = outer(u, v)
    A2 = as_tensor(A[i, j], (i, j))
    assert A2 == A

    Bij = u[i] * v[j]
    Bij2 = as_tensor(Bij, (i, j))[i, j]
    as_tensor(Bij, (i, j))
    assert Bij2 == Bij


@pytest.mark.parametrize("shape", [(3,), (3, 2)], ids=("vector", "matrix"))
def test_tensor_from_indexed(self, shape):
    element = FiniteElement("Lagrange", triangle, 1, shape, identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    assert as_tensor(reshape([f[i] for i in ndindex(f.ufl_shape)], f.ufl_shape).tolist()) is f


def test_nested_indexed(self):
    # Test that a nested Indexed expression simplifies to the existing Indexed object
    shape = (2,)
    element = FiniteElement("Lagrange", triangle, 1, shape, identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)

    comps = tuple(f[i] for i in range(2))
    assert all(isinstance(c, Indexed) for c in comps)
    expr = as_tensor(list(reversed(comps)))

    multiindex = MultiIndex((FixedIndex(0),))
    assert Indexed(expr, multiindex) is expr[0]
    assert Indexed(expr, multiindex) is comps[1]


def test_repeated_indexing(self):
    # Test that an Indexed with repeated indices does not contract indices
    shape = (2, 2)
    element = FiniteElement("Lagrange", triangle, 1, shape, identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    x = Coefficient(space)
    C = as_tensor([x, x])

    fi = FixedIndex(0)
    i = Index()
    ii = MultiIndex((fi, i, i))
    expr = Indexed(C, ii)
    assert i.count() in expr.ufl_free_indices
    assert isinstance(expr, Indexed)
    B, jj = expr.ufl_operands
    assert B is x
    assert tuple(jj) == tuple(ii[1:])


def test_untangle_indexed_component_tensor(self):
    shape = (2, 2, 2, 2)
    element = FiniteElement("Lagrange", triangle, 1, shape, identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    C = Coefficient(space)

    r = len(shape)
    kk = indices(r)

    # Untangle as_tensor(C[kk], kk) -> C
    B = as_tensor(Indexed(C, MultiIndex(kk)), kk)
    assert B is C

    # Untangle as_tensor(C[kk], jj)[ii] -> C[ll]
    jj = kk[2:]
    A = as_tensor(Indexed(C, MultiIndex(kk)), jj)
    assert A is not C

    ii = kk
    expr = Indexed(A, MultiIndex(ii))
    assert isinstance(expr, Indexed)
    B, ll = expr.ufl_operands
    assert B is C

    rep = dict(zip(jj, ii))
    expected = tuple(rep.get(k, k) for k in kk)
    assert tuple(ll) == expected


def test_simplify_indexed(self):
    element = FiniteElement("Lagrange", triangle, 1, (3,), identity_pullback, H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2,), identity_pullback, H1))
    space = FunctionSpace(domain, element)
    u = Coefficient(space)
    z = Zero(())
    i = Index()
    j = Index()
    # ListTensor
    lt = ListTensor(z, z, u[1])
    assert Indexed(lt, MultiIndex((FixedIndex(2),))) == u[1]
    # ListTensor -- nested
    l0 = ListTensor(z, u[1], z)
    l1 = ListTensor(z, z, u[2])
    l2 = ListTensor(u[0], z, z)
    ll = ListTensor(l0, l1, l2)
    assert Indexed(ll, MultiIndex((FixedIndex(1), FixedIndex(2)))) == u[2]
    assert Indexed(ll, MultiIndex((FixedIndex(2), i))) == l2[i]
    # ComponentTensor + ListTensor
    c = ComponentTensor(Indexed(ll, MultiIndex((i, j))), MultiIndex((j, i)))
    assert Indexed(c, MultiIndex((FixedIndex(1), FixedIndex(2)))) == l2[1]
