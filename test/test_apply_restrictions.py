from pytest import raises
from utils import FiniteElement, LagrangeElement

from ufl import (
    Coefficient,
    FacetNormal,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    as_tensor,
    grad,
    i,
    triangle,
)
from ufl.algorithms.apply_restrictions import apply_restrictions
from ufl.algorithms.renumbering import renumber_indices
from ufl.pullback import identity_pullback
from ufl.sobolevspace import L2


def test_apply_restrictions():
    cell = triangle
    V0 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, L2)
    V1 = LagrangeElement(cell, 1)
    V2 = LagrangeElement(cell, 2)

    domain = Mesh(LagrangeElement(cell, 1, (2,)))
    v0_space = FunctionSpace(domain, V0)
    v1_space = FunctionSpace(domain, V1)
    v2_space = FunctionSpace(domain, V2)

    f0 = Coefficient(v0_space)
    f = Coefficient(v1_space)
    g = Coefficient(v2_space)
    n = FacetNormal(domain)
    x = SpatialCoordinate(domain)

    assert raises(BaseException, lambda: apply_restrictions(f0))
    assert raises(BaseException, lambda: apply_restrictions(grad(f)))
    assert raises(BaseException, lambda: apply_restrictions(n))

    # Continuous function gets default restriction if none
    # provided otherwise the user choice is respected
    assert apply_restrictions(f) == f("+")
    assert apply_restrictions(f("-")) == f("-")
    assert apply_restrictions(f("+")) == f("+")

    # Propagation to terminals
    assert apply_restrictions((f + f0)("+")) == f("+") + f0("+")

    # Propagation stops at grad
    assert apply_restrictions(grad(f)("-")) == grad(f)("-")
    assert apply_restrictions((grad(f) ** 2)("+")) == grad(f)("+") ** 2
    assert apply_restrictions((grad(f) + grad(g))("-")) == (grad(f)("-") + grad(g)("-"))

    # x is the same from both sides but computed from one of them
    assert apply_restrictions(x) == x("+")

    # n on a linear mesh is opposite pointing from the other side
    assert apply_restrictions(n("+")) == n("+")
    assert renumber_indices(apply_restrictions(n("-"))) == renumber_indices(
        as_tensor(-1 * n("+")[i], i)
    )
    # This would be nicer, but -f is translated to -1*f which is
    # translated to as_tensor(-1*f[i], i). assert
    # apply_restrictions(n('-')) == -n('+')
