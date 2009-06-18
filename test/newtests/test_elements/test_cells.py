
from ufl import Cell, interval, triangle, tetrahedron, quadrilateral, hexahedron

def construct_cells(domain, degree, space):
    # Construct linear cell
    cell = Cell(domain)
    assert cell == eval(domain)

    # Construct higher order cell
    cell = Cell(domain, degree)

    # Construct cell in other space
    if cell.topological_dimension() >= space.dimension():
        cell = Cell(domain, degree, space)
    else:
        failed = True
        try:
            cell = Cell(domain, degree, space)
            failed = False
        except:
            pass
        assert failed

def test_cells():
    from ufl import R1, R2, R3
    for domain in ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"):
        for d in range(1, 3):
            for s in (R1, R2, R3):
                yield construct_cells, domain, d, s

