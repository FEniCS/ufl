"Utility objects for pretty syntax in user code."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2009-01-09"

# Modified by Anders Logg, 2008

from ufl.integral import Integral
from ufl.indexing import Index
from ufl.geometry import Cell

# Default indices
i, j, k, l = [Index() for _i in range(4)]
p, q, r, s = [Index() for _i in range(4)]

# Default integrals
dx = Integral(Integral.CELL, 0)
ds = Integral(Integral.EXTERIOR_FACET, 0)
dS = Integral(Integral.INTERIOR_FACET, 0)

# Cell types
interval      = Cell("interval", 1)
triangle      = Cell("triangle", 1)
tetrahedron   = Cell("tetrahedron", 1)
quadrilateral = Cell("quadrilateral", 1)
hexahedron    = Cell("hexahedron", 1)
