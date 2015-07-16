#!/usr/bin/env py.test

import pytest

from ufl import *

def test_reference_shapes():
    #show_elements()

    cell = Cell("triangle", 3)

    V = FiniteElement("N1curl", cell, 1)
    assert V.value_shape() == (3,)
    assert V.reference_value_shape() == (2,)

    U = FiniteElement("RT", cell, 1)
    assert U.value_shape() == (3,)
    assert U.reference_value_shape() == (2,)

    W = FiniteElement("CG", cell, 1)
    assert W.value_shape() == ()
    assert W.reference_value_shape() == ()

    Q = VectorElement("CG", cell, 1)
    assert Q.value_shape() == (3,)
    assert Q.reference_value_shape() == (3,)

    T = TensorElement("CG", cell, 1)
    assert T.value_shape() == (3, 3)
    assert T.reference_value_shape() == (3,3)

    S = TensorElement("CG", cell, 1, symmetry=True)
    assert S.value_shape() == (3, 3)
    assert S.reference_value_shape() == (6,)

    M = MixedElement(V, U, W)
    assert M.value_shape() == (7,)
    assert M.reference_value_shape() == (5,)
