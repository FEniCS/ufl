import ufl
import pytest


def test_interval():
    cell = ufl.interval
    assert cell.num_vertices() == 2
    assert cell.num_edges() == 1
    assert cell.num_faces() == 0


def test_triangle():
    cell = ufl.triangle
    assert cell.num_vertices() == 3
    assert cell.num_edges() == 3
    assert cell.num_faces() == 1


def test_quadrilateral():
    cell = ufl.quadrilateral
    assert cell.num_vertices() == 4
    assert cell.num_edges() == 4
    assert cell.num_faces() == 1


def test_tetrahedron():
    cell = ufl.tetrahedron
    assert cell.num_vertices() == 4
    assert cell.num_edges() == 6
    assert cell.num_faces() == 4


def test_hexahedron():
    cell = ufl.hexahedron
    assert cell.num_vertices() == 8
    assert cell.num_edges() == 12
    assert cell.num_faces() == 6


def test_pentatope():
    cell = ufl.pentatope
    assert cell.num_vertices() == 5
    assert cell.num_edges() == 10
    assert cell.num_faces() == 10


def test_tesseract():
    cell = ufl.tesseract
    assert cell.num_vertices() == 16
    assert cell.num_edges() == 32
    assert cell.num_faces() == 24



@pytest.mark.parametrize("cell", [ufl.interval])
def test_cells_1d(cell):
    assert cell.num_facets() == cell.num_vertices()
    assert cell.num_ridges() == 0
    assert cell.num_peaks() == 0


@pytest.mark.parametrize("cell", [ufl.triangle, ufl.quadrilateral])
def test_cells_2d(cell):
    assert cell.num_facets() == cell.num_edges()
    assert cell.num_ridges() == cell.num_vertices()
    assert cell.num_peaks() == 0


@pytest.mark.parametrize("cell", [ufl.tetrahedron, ufl.hexahedron, ufl.prism, ufl.pyramid])
def test_cells_3d(cell):
    assert cell.num_facets() == cell.num_faces()
    assert cell.num_ridges() == cell.num_edges()
    assert cell.num_peaks() == cell.num_vertices()


@pytest.mark.parametrize("cell", [ufl.tesseract, ufl.pentatope])
def test_cells_4d(cell):
    assert cell.num_facets() == cell.num_sub_entities(3)
    assert cell.num_ridges() == cell.num_faces()
    assert cell.num_peaks() == cell.num_edges()


def test_tensorproductcell():
    orig = ufl.TensorProductCell(ufl.interval, ufl.interval)
    cell = orig.reconstruct()
    assert cell.sub_cells() == orig.sub_cells()
    assert cell.topological_dimension() == orig.topological_dimension()
    assert cell.geometric_dimension() == orig.geometric_dimension()
