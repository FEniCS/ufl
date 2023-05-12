#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

from pytest import raises
from ufl_legacy import *
from ufl_legacy.algorithms.apply_restrictions import apply_restrictions, apply_default_restrictions
from ufl_legacy.algorithms.renumbering import renumber_indices


def test_apply_restrictions():
    cell = triangle
    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)
    V2 = FiniteElement("Lagrange", cell, 2)
    f0 = Coefficient(V0)
    f = Coefficient(V1)
    g = Coefficient(V2)
    n = FacetNormal(cell)
    x = SpatialCoordinate(cell)

    assert raises(UFLException, lambda: apply_restrictions(f0))
    assert raises(UFLException, lambda: apply_restrictions(grad(f)))
    assert raises(UFLException, lambda: apply_restrictions(n))

    # Continuous function gets default restriction if none
    # provided otherwise the user choice is respected
    assert apply_restrictions(f) == f('+')
    assert apply_restrictions(f('-')) == f('-')
    assert apply_restrictions(f('+')) == f('+')

    # Propagation to terminals
    assert apply_restrictions((f + f0)('+')) == f('+') + f0('+')

    # Propagation stops at grad
    assert apply_restrictions(grad(f)('-')) == grad(f)('-')
    assert apply_restrictions((grad(f)**2)('+')) == grad(f)('+')**2
    assert apply_restrictions((grad(f) + grad(g))('-')) == (grad(f)('-') + grad(g)('-'))

    # x is the same from both sides but computed from one of them
    assert apply_default_restrictions(x) == x('+')

    # n on a linear mesh is opposite pointing from the other side
    assert apply_restrictions(n('+')) == n('+')
    assert renumber_indices(apply_restrictions(n('-'))) == renumber_indices(as_tensor(-1*n('+')[i], i))
    # This would be nicer, but -f is translated to -1*f which is translated to as_tensor(-1*f[i], i).
    # assert apply_restrictions(n('-')) == -n('+')
