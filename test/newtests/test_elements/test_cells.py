
def test_cells():
    from ufl import Cell
    for domain in ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"):
        for d in range(1, 3):
            cell = Cell(domain)
            cell = Cell(domain, d)
            cell = Cell(domain, d, 3)

