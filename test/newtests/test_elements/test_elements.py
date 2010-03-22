
# TODO: check more element definitions, mixed elements, subelements, element restrictions, element unions

from ufl import FiniteElement, VectorElement, TensorElement, MixedElement, EnrichedElement, ElementRestriction
from ufl import interval, triangle, quadrilateral, tetrahedron, hexahedron

all_cells = (interval, triangle, quadrilateral, tetrahedron, hexahedron)

def construct_element(name, domain, degree, value_rank):
    element = FiniteElement(name, domain, degree)
    assert value_rank == len(element.value_shape())

def construct_vector_element(name, domain, degree, value_rank):
    element = VectorElement(name, domain, degree)
    assert value_rank+1 == len(element.value_shape())

def construct_tensor_element(name, domain, degree, value_rank):
    element = TensorElement(name, domain, degree)
    assert value_rank+2 == len(element.value_shape())

def test_element_construction():
    "Iterate over all registered elements and try to construct instances."
    from ufl.elementlist import ufl_elements
    for k in sorted(ufl_elements.keys()):
        (family, short_name, value_rank, degree_range, domains) = ufl_elements[k]
        if degree_range:
            a, b = degree_range
            if b is None:
                b = a + 2
        for name in (family, short_name):
            for domain in domains:
                for degree in range(a, b):
                    yield construct_element, name, domain, degree, value_rank
                    yield construct_vector_element, name, domain, degree, value_rank
                    yield construct_tensor_element, name, domain, degree, value_rank


def test_vector_galerkin():
    for cell in all_cells:
        dim = cell.geometric_dimension()
        for p in range(1,10):
            for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                element = VectorElement(family, cell, p)
                assert (element.value_shape() == (dim,))
                for i in range(dim):
                    c = element.extract_component(i)
                    assert (c[0] == ())

def test_tensor_galerkin():
    for cell in all_cells:
        dim = cell.geometric_dimension()
        for p in range(1,10):
            for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                element = TensorElement(family, cell, p)
                assert (element.value_shape() == (dim,dim))
                for i in range(dim):
                    for j in range(dim):
                        c = element.extract_component((i,j))
                        assert (c[0] == ())

def test_tensor_symmetry():
    for cell in all_cells:
        dim = cell.geometric_dimension()
        for p in range(1,10):
            for s in (None, True, {(0,1): (1,0)}):
                for family in ("Lagrange", "CG", "Discontinuous Lagrange", "DG"):
                    if isinstance(s, dict):
                        element = TensorElement(family, cell, p, shape=(dim,dim), symmetry=s)
                    else:
                        element = TensorElement(family, cell, p, symmetry=s)
                    assert (element.value_shape() == (dim,dim))
                    for i in range(dim):
                        for j in range(dim):
                            c = element.extract_component((i,j))
                            assert (c[0] == ())

def test_bdm():
    for cell in (triangle, tetrahedron):
        dim = cell.geometric_dimension()
        element = FiniteElement("BDM", cell, 1)
        assert (element.value_shape() == (dim,))

def test_vector_bdm():
    for cell in (triangle, tetrahedron):
        dim = cell.geometric_dimension()
        element = VectorElement("BDM", cell, 1)
        assert (element.value_shape() == (dim,dim))

def test_mixed():
    for cell in (triangle, tetrahedron):
        dim = cell.geometric_dimension()
        velement = VectorElement("CG", cell, 2)
        pelement = FiniteElement("CG", cell, 1)
        TH1 = MixedElement(velement, pelement)
        TH2 = velement * pelement
        assert ( repr(TH1) == repr(TH2) )
        assert ( TH1.value_shape() == (dim+1,) )
        assert ( TH2.value_shape() == (dim+1,) )

