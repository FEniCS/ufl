from ufl import Cell, Mesh
from ufl.finiteelement import FiniteElement, MixedElement, SymmetricElement
from ufl.functionspace import FunctionSpace
from ufl.pullback import contravariant_piola, covariant_piola, identity_pullback
from ufl.sobolevspace import H1, HCurl, HDiv


def test_reference_shapes():
    # show_elements()
    cell = Cell("triangle")
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (3,), identity_pullback, H1))

    V = FiniteElement("N1curl", cell, 1, (2,), covariant_piola, HCurl)
    Vspace = FunctionSpace(domain, V)
    assert Vspace.value_shape == (3,)
    assert V.reference_value_shape == (2,)

    U = FiniteElement("Raviart-Thomas", cell, 1, (2,), contravariant_piola, HDiv)
    Uspace = FunctionSpace(domain, U)
    assert Uspace.value_shape == (3,)
    assert U.reference_value_shape == (2,)

    W = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    Wspace = FunctionSpace(domain, W)
    assert Wspace.value_shape == ()
    assert W.reference_value_shape == ()

    Q = FiniteElement("Lagrange", cell, 1, (3,), identity_pullback, H1)
    Qspace = FunctionSpace(domain, Q)
    assert Qspace.value_shape == (3,)
    assert Q.reference_value_shape == (3,)

    T = FiniteElement("Lagrange", cell, 1, (3, 3), identity_pullback, H1)
    Tspace = FunctionSpace(domain, T)
    assert Tspace.value_shape == (3, 3)
    assert T.reference_value_shape == (3, 3)

    S = SymmetricElement(
        {
            (0, 0): 0,
            (1, 0): 1,
            (2, 0): 2,
            (0, 1): 1,
            (1, 1): 3,
            (2, 1): 4,
            (0, 2): 2,
            (1, 2): 4,
            (2, 2): 5,
        },
        [FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1) for _ in range(6)],
    )
    Sspace = FunctionSpace(domain, S)
    assert Sspace.value_shape == (3, 3)
    assert S.reference_value_shape == (6,)

    M = MixedElement([V, U, W])
    Mspace = FunctionSpace(domain, M)
    assert Mspace.value_shape == (7,)
    assert M.reference_value_shape == (5,)
