from utils import FiniteElement, LagrangeElement

from ufl import (
    CellDiameter,
    CellVolume,
    Circumradius,
    FacetArea,
    FacetNormal,
    FunctionSpace,
    Index,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_matrix,
    as_ufl,
    as_vector,
    quadrilateral,
    tetrahedron,
    triangle,
)
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1


def test_str_int_value(self):
    assert str(as_ufl(3)) == "3"


def test_str_float_value(self):
    assert str(as_ufl(3.14)) == "3.14"


def test_str_zero(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    x = SpatialCoordinate(domain)
    assert str(as_ufl(0)) == "0"
    assert str(0 * x) == "0 (shape (2,))"
    assert str(0 * x * x[Index(42)]) == "0 (shape (2,), index labels (42,))"


def test_str_index(self):
    assert str(Index(3)) == "i_3"
    assert str(Index(42)) == "i_{42}"


def test_str_coordinate(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    assert str(SpatialCoordinate(domain)) == "x"
    assert str(SpatialCoordinate(domain)[0]) == "x[0]"


def test_str_normal(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    assert str(FacetNormal(domain)) == "n"
    assert str(FacetNormal(domain)[0]) == "n[0]"


def test_str_circumradius(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    assert str(Circumradius(domain)) == "circumradius"


def test_str_diameter(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    assert str(CellDiameter(domain)) == "diameter"


def test_str_facetarea(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    assert str(FacetArea(domain)) == "facetarea"


def test_str_volume(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    assert str(CellVolume(domain)) == "volume"


def test_str_scalar_argument(self):
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    v = TestFunction(FunctionSpace(domain, LagrangeElement(triangle, 1)))
    u = TrialFunction(FunctionSpace(domain, LagrangeElement(triangle, 1)))
    assert str(v) == "v_0"
    assert str(u) == "v_1"


# def test_str_vector_argument(self): # FIXME

# def test_str_scalar_coefficient(self): # FIXME

# def test_str_vector_coefficient(self): # FIXME


def test_str_list_vector():
    domain = Mesh(LagrangeElement(tetrahedron, 1, (3,)))
    x, y, z = SpatialCoordinate(domain)
    v = as_vector((z, y, x))
    assert str(v) == (f"[{z}, {y}, {x}]")


def test_str_list_vector_with_zero():
    domain = Mesh(LagrangeElement(tetrahedron, 1, (3,)))
    x, y, z = SpatialCoordinate(domain)
    v = as_vector((x, 0, 0))
    assert str(v) == (f"[{x}, 0, 0]")


def test_str_list_matrix():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    x, y = SpatialCoordinate(domain)
    v = as_matrix(((2 * x, 3 * y), (4 * x, 5 * y)))
    a = str(2 * x)
    b = str(3 * y)
    c = str(4 * x)
    d = str(5 * y)
    assert str(v) == (f"[\n  [{a}, {b}],\n  [{c}, {d}]\n]")


def test_str_list_matrix_with_zero():
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    x, y = SpatialCoordinate(domain)
    v = as_matrix(((2 * x, 3 * y), (0, 0)))
    a = str(2 * x)
    b = str(3 * y)
    c = str(as_vector((0, 0)))
    assert str(v) == (f"[\n  [{a}, {b}],\n{c}\n]")


# FIXME: Add more tests for tensors collapsing
#        partly or completely into Zero!


def test_str_element():
    elem = FiniteElement("Q", quadrilateral, 1, (), identity_pullback, H1)
    assert repr(elem) == 'utils.FiniteElement("Q", quadrilateral, 1, (), IdentityPullback(), H1)'
    assert str(elem) == "<Q1 on a quadrilateral>"
