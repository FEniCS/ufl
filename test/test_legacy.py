from ufl import (Coefficient, FunctionSpace, H1, Mesh,
                 dx, hexahedron, identity_pull_back, inner, interval, quadrilateral, tetrahedron, triangle)
from ufl.legacy import FiniteElement, MixedElement, TensorElement, VectorElement, WithMapping

all_cells = (interval, triangle, tetrahedron, quadrilateral, hexahedron)


def test_legacy_vs_new():
    from ufl.finiteelement import FiniteElement as NewFiniteElement
    e = FiniteElement("Lagrange", triangle, 1)
    new_e = NewFiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    assert e.sobolev_space == new_e.sobolev_space
    assert e.pull_back == new_e.pull_back
    assert e.embedded_superdegree == new_e.embedded_superdegree
    assert e.embedded_subdegree == new_e.embedded_subdegree
    assert e.cell == new_e.cell
    assert e.reference_value_shape == new_e.reference_value_shape
    assert e.value_shape == new_e.value_shape
    assert e.reference_value_size == new_e.reference_value_size
    assert e.value_size == new_e.value_size
    assert e.num_sub_elements == new_e.num_sub_elements
    assert e.sub_elements == new_e.sub_elements
    assert e.is_cellwise_constant() == new_e.is_cellwise_constant()


def test_scalar_galerkin():
    for cell in all_cells:
        for p in range(1, 10):
            for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG", "Discontinuous Lagrange L2", "DG L2"):
                element = FiniteElement(family, cell, p)
                assert element.value_shape == ()
                assert element == eval(repr(element))
    for p in range(1, 10):
        for family in ("TDG", "Discontinuous Taylor"):
            element = FiniteElement(family, interval, p)
            assert element.value_shape == ()


def test_vector_galerkin():
    for cell in all_cells:
        dim = cell.geometric_dimension()
        # shape = () if dim == 1 else (dim,)
        shape = (dim,)
        for p in range(1, 10):
            for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG", "Discontinuous Lagrange L2", "DG L2"):
                element = VectorElement(family, cell, p)
                assert element.value_shape == shape
                assert element == eval(repr(element))
                for i in range(dim):
                    c = element.extract_component(i)
                    assert c[0] == ()


def test_tensor_galerkin():
    for cell in all_cells:
        dim = cell.geometric_dimension()
        # shape = () if dim == 1 else (dim,dim)
        shape = (dim, dim)
        for p in range(1, 10):
            for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG", "Discontinuous Lagrange L2", "DG L2"):
                element = TensorElement(family, cell, p)
                assert element.value_shape == shape
                assert element == eval(repr(element))
                for i in range(dim):
                    for j in range(dim):
                        c = element.extract_component((i, j))
                        assert c[0] == ()


def test_tensor_symmetry():
    for cell in all_cells:
        dim = cell.geometric_dimension()
        for p in range(1, 10):
            for s in (None, True, {(0, 1): (1, 0)}):
                # Symmetry dict is invalid for interval cell
                if isinstance(s, dict) and cell == interval:
                    continue

                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG", "Discontinuous Lagrange L2", "DG L2"):
                    if isinstance(s, dict):
                        element = TensorElement(
                            family, cell, p, shape=(dim, dim), symmetry=s)
                    else:
                        element = TensorElement(family, cell, p, symmetry=s)
                    assert element.value_shape, (dim == dim)
                    assert element == eval(repr(element))
                    for i in range(dim):
                        for j in range(dim):
                            c = element.extract_component((i, j))
                            assert c[0] == ()


def test_mixed_tensor_symmetries():
    from ufl.algorithms import expand_compounds, expand_indices

    S = FiniteElement('CG', triangle, 1)
    V = VectorElement('CG', triangle, 1)
    T = TensorElement('CG', triangle, 1, symmetry=True)

    print(T.pull_back)

    # M has dimension 4+1, symmetries are 2->1
    M = T * S
    domain = Mesh(VectorElement("Lagrange", triangle, 1))
    m_space = FunctionSpace(domain, M)
    P = Coefficient(m_space)
    M = inner(P, P) * dx

    M2 = expand_indices(expand_compounds(M))
    assert '[1]' in str(M2)
    assert '[2]' not in str(M2)

    # M has dimension 2+(1+4), symmetries are 5->4
    M = V * (S * T)
    m_space = FunctionSpace(domain, M)
    P = Coefficient(m_space)
    M = inner(P, P) * dx

    M2 = expand_indices(expand_compounds(M))
    assert '[4]' in str(M2)
    assert '[5]' not in str(M2)


def test_bdm():
    for cell in (triangle, tetrahedron):
        dim = cell.geometric_dimension()
        element = FiniteElement("BDM", cell, 1)
        assert element.value_shape == (dim,)
        assert element == eval(repr(element))


def test_vector_bdm():
    for cell in (triangle, tetrahedron):
        dim = cell.geometric_dimension()
        element = VectorElement("BDM", cell, 1)
        assert element.value_shape, (dim == dim)
        assert element == eval(repr(element))


def test_mtw():
    cell = triangle
    element = FiniteElement("MTW", cell, 3)
    assert element.value_shape == (cell.geometric_dimension(), )
    assert element == eval(repr(element))
    assert element.mapping() == "contravariant Piola"


def test_mixed():
    for cell in (triangle, tetrahedron):
        dim = cell.geometric_dimension()
        velement = VectorElement("CG", cell, 2)
        pelement = FiniteElement("CG", cell, 1)
        TH1 = MixedElement(velement, pelement)
        TH2 = velement * pelement
        assert TH1.value_shape == (dim + 1,)
        assert TH2.value_shape == (dim + 1,)
        assert repr(TH1) == repr(TH2)
        assert TH1 == eval(repr(TH2))
        assert TH2 == eval(repr(TH1))


def test_nested_mixed():
    for cell in (triangle, tetrahedron):
        dim = cell.geometric_dimension()
        velement = VectorElement("CG", cell, 2)
        pelement = FiniteElement("CG", cell, 1)
        TH1 = MixedElement((velement, pelement), pelement)
        TH2 = velement * pelement * pelement
        assert TH1.value_shape == (dim + 2,)
        assert TH2.value_shape == (dim + 2,)
        assert repr(TH1) == repr(TH2)
        assert TH1 == eval(repr(TH2))
        assert TH2 == eval(repr(TH1))


def test_quadrature_scheme():
    for cell in (triangle, tetrahedron):
        for q in (None, 1, 2, 3):
            element = FiniteElement("CG", cell, 1, quad_scheme=q)
            assert element.quadrature_scheme() == q
            assert element == eval(repr(element))


def test_missing_cell():
    # These special cases are here to allow missing
    # cell in PyDOLFIN Constant and Expression
    for cell in (triangle, None):
        element = FiniteElement("Real", cell, 0)
        assert element == eval(repr(element))
        element = FiniteElement("Undefined", cell, None)
        assert element == eval(repr(element))
        element = VectorElement("Lagrange", cell, 1, dim=2)
        assert element == eval(repr(element))
        element = TensorElement("DG", cell, 1, shape=(2, 2))
        assert element == eval(repr(element))
        element = TensorElement("DG L2", cell, 1, shape=(2, 2))
        assert element == eval(repr(element))


def test_invalid_degree():
    cell = triangle
    for degree in (1, None):
        element = FiniteElement("CG", cell, degree)
        assert element == eval(repr(element))
        element = VectorElement("CG", cell, degree)
        assert element == eval(repr(element))


def test_lobatto():
    cell = interval
    for degree in (1, 2, None):
        element = FiniteElement("Lob", cell, degree)
        assert element == eval(repr(element))

        element = FiniteElement("Lobatto", cell, degree)
        assert element == eval(repr(element))


def test_radau():
    cell = interval
    for degree in (0, 1, 2, None):
        element = FiniteElement("Rad", cell, degree)
        assert element == eval(repr(element))

        element = FiniteElement("Radau", cell, degree)
        assert element == eval(repr(element))


def test_mse():
    for degree in (2, 3, 4, 5):
        element = FiniteElement('EGL', interval, degree)
        assert element == eval(repr(element))

        element = FiniteElement('EGL-Edge', interval, degree - 1)
        assert element == eval(repr(element))

        element = FiniteElement('EGL-Edge L2', interval, degree - 1)
        assert element == eval(repr(element))

    for degree in (1, 2, 3, 4, 5):
        element = FiniteElement('GLL', interval, degree)
        assert element == eval(repr(element))

        element = FiniteElement('GLL-Edge', interval, degree - 1)
        assert element == eval(repr(element))

        element = FiniteElement('GLL-Edge L2', interval, degree - 1)
        assert element == eval(repr(element))


def test_withmapping():
    base = FiniteElement("CG", interval, 1)
    element = WithMapping(base, "identity")
    assert element == eval(repr(element))
