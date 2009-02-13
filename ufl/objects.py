"Utility objects for pretty syntax in user code."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-02-13"

# Modified by Anders Logg, 2008

from ufl.indexing import indices
from ufl.integral import Measure
from ufl.geometry import Cell

# Default indices
i, j, k, l = indices(4)
p, q, r, s = indices(4)

# Default measures for integration
dx = Measure(Measure.CELL, 0)
ds = Measure(Measure.EXTERIOR_FACET, 0)
dS = Measure(Measure.INTERIOR_FACET, 0)

# Cell types
interval      = Cell("interval", 1)
triangle      = Cell("triangle", 1)
tetrahedron   = Cell("tetrahedron", 1)
quadrilateral = Cell("quadrilateral", 1)
hexahedron    = Cell("hexahedron", 1)
