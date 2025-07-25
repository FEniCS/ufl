"""Tests of the change to local representaiton algorithms."""

from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, as_tensor, grad, indices, triangle
from ufl.algorithms import change_to_reference_grad
from ufl.algorithms.renumbering import renumber_indices
from ufl.classes import JacobianInverse, ReferenceGrad


def test_change_to_reference_grad():
    cell = triangle
    domain = Mesh(LagrangeElement(cell, 1, (2,)))
    U = FunctionSpace(domain, LagrangeElement(cell, 1))
    V = FunctionSpace(domain, LagrangeElement(cell, 1, (2,)))
    u = Coefficient(U)
    v = Coefficient(V)
    Jinv = JacobianInverse(domain)
    i, j, k = indices(3)
    q, r, s = indices(3)
    (t,) = indices(1)

    # Single grad change on a scalar function
    expr = grad(u)
    actual = change_to_reference_grad(expr)
    expected = as_tensor(Jinv[k, i] * ReferenceGrad(u)[k], (i,))
    assert renumber_indices(actual) == renumber_indices(expected)

    # Single grad change on a vector valued function
    expr = grad(v)
    actual = change_to_reference_grad(expr)
    expected = as_tensor(Jinv[k, j] * ReferenceGrad(v)[i, k], (i, j))
    assert renumber_indices(actual) == renumber_indices(expected)

    # Multiple grads should work fine for affine domains:
    expr = grad(grad(u))
    actual = change_to_reference_grad(expr)
    expected = as_tensor(Jinv[s, j] * (Jinv[r, i] * ReferenceGrad(ReferenceGrad(u))[r, s]), (i, j))
    assert renumber_indices(actual) == renumber_indices(expected)

    expr = grad(grad(grad(u)))
    actual = change_to_reference_grad(expr)
    expected = as_tensor(
        Jinv[s, k]
        * (Jinv[r, j] * (Jinv[q, i] * ReferenceGrad(ReferenceGrad(ReferenceGrad(u)))[q, r, s])),
        (i, j, k),
    )
    assert renumber_indices(actual) == renumber_indices(expected)

    # Multiple grads on a vector valued function
    expr = grad(grad(v))
    actual = change_to_reference_grad(expr)
    expected = as_tensor(
        Jinv[s, j] * (Jinv[r, i] * ReferenceGrad(ReferenceGrad(v))[t, r, s]), (t, i, j)
    )
    assert renumber_indices(actual) == renumber_indices(expected)

    expr = grad(grad(grad(v)))
    actual = change_to_reference_grad(expr)
    expected = as_tensor(
        Jinv[s, k]
        * (Jinv[r, j] * (Jinv[q, i] * ReferenceGrad(ReferenceGrad(ReferenceGrad(v)))[t, q, r, s])),
        (t, i, j, k),
    )
    assert renumber_indices(actual) == renumber_indices(expected)

    # print tree_format(expected)
    # print tree_format(actual)
    # print tree_format(renumber_indices(actual))
    # print tree_format(renumber_indices(expected))
