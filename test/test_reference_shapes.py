from ufl.finiteelement import FiniteElement, MixedElement
from ufl.sobolevspace import H1, HCurl, HDiv
from ufl import Cell


def test_reference_shapes():
    # show_elements()

    cell = Cell("triangle", 3)

    V = FiniteElement("N1curl", cell, 1, (3, ), (2, ), "covariant Piola", HCurl)
    assert V.value_shape == (3,)
    assert V.reference_value_shape == (2,)

    U = FiniteElement("Raviart-Thomas", cell, 1, (3, ), (2, ), "contravariant Piola", HDiv)
    assert U.value_shape == (3,)
    assert U.reference_value_shape == (2,)

    W = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    assert W.value_shape == ()
    assert W.reference_value_shape == ()

    Q = FiniteElement("Lagrange", cell, 1, (3, ), (3, ), "identity", H1)
    assert Q.value_shape == (3,)
    assert Q.reference_value_shape == (3,)

    T = FiniteElement("Lagrange", cell, 1, (3, 3), (3, 3), "identity", H1)
    assert T.value_shape == (3, 3)
    assert T.reference_value_shape == (3, 3)

    S = FiniteElement("Lagrange", cell, 1, (3, 3), (6, ), "identity", H1, component_map={
        (0, 0): 0, (1, 0): 1, (2, 0): 2, (0, 1): 1, (1, 1): 3, (2, 1): 4, (0, 2): 2, (1, 2): 4, (2, 2): 5})
    assert S.value_shape == (3, 3)
    assert S.reference_value_shape == (6,)

    M = MixedElement([V, U, W])
    assert M.value_shape == (7,)
    assert M.reference_value_shape == (5,)
