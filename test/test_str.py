from ufl import (CellDiameter, CellVolume, Circumradius, FacetArea, FacetNormal, FiniteElement, Index,
                 SpatialCoordinate, TestFunction, TrialFunction, as_matrix, as_ufl, as_vector, quadrilateral,
                 tetrahedron, triangle)


def test_str_int_value(self):
    assert str(as_ufl(3)) == "3"


def test_str_float_value(self):
    assert str(as_ufl(3.14)) == "3.14"


def test_str_zero(self):
    x = SpatialCoordinate(triangle)
    assert str(as_ufl(0)) == "0"
    assert str(0*x) == "0 (shape (2,))"
    assert str(0*x*x[Index(42)]) == "0 (shape (2,), index labels (42,))"


def test_str_index(self):
    assert str(Index(3)) == "i_3"
    assert str(Index(42)) == "i_{42}"


def test_str_coordinate(self):
    assert str(SpatialCoordinate(triangle)) == "x"
    assert str(SpatialCoordinate(triangle)[0]) == "x[0]"


def test_str_normal(self):
    assert str(FacetNormal(triangle)) == "n"
    assert str(FacetNormal(triangle)[0]) == "n[0]"


def test_str_circumradius(self):
    assert str(Circumradius(triangle)) == "circumradius"


def test_str_diameter(self):
    assert str(CellDiameter(triangle)) == "diameter"


# def test_str_cellsurfacearea(self):
#     assert str(CellSurfaceArea(triangle)) == "surfacearea"


def test_str_facetarea(self):
    assert str(FacetArea(triangle)) == "facetarea"


def test_str_volume(self):
    assert str(CellVolume(triangle)) == "volume"


def test_str_scalar_argument(self):
    v = TestFunction(FiniteElement("CG", triangle, 1))
    u = TrialFunction(FiniteElement("CG", triangle, 1))
    assert str(v) == "v_0"
    assert str(u) == "v_1"


# def test_str_vector_argument(self): # FIXME

# def test_str_scalar_coefficient(self): # FIXME

# def test_str_vector_coefficient(self): # FIXME


def test_str_list_vector():
    x, y, z = SpatialCoordinate(tetrahedron)
    v = as_vector((x, y, z))
    assert str(v) == ("[%s, %s, %s]" % (x, y, z))


def test_str_list_vector_with_zero():
    x, y, z = SpatialCoordinate(tetrahedron)
    v = as_vector((x, 0, 0))
    assert str(v) == ("[%s, 0, 0]" % (x,))


def test_str_list_matrix():
    x, y = SpatialCoordinate(triangle)
    v = as_matrix(((2*x, 3*y),
                   (4*x, 5*y)))
    a = str(2*x)
    b = str(3*y)
    c = str(4*x)
    d = str(5*y)
    assert str(v) == ("[\n  [%s, %s],\n  [%s, %s]\n]" % (a, b, c, d))


def test_str_list_matrix_with_zero():
    x, y = SpatialCoordinate(triangle)
    v = as_matrix(((2*x, 3*y),
                   (0, 0)))
    a = str(2*x)
    b = str(3*y)
    c = str(as_vector((0, 0)))
    assert str(v) == ("[\n  [%s, %s],\n%s\n]" % (a, b, c))


# FIXME: Add more tests for tensors collapsing
#        partly or completely into Zero!


def test_str_element():
    elem = FiniteElement("Q", quadrilateral, 1)
    assert str(elem) == "<Q1 on a quadrilateral>"
