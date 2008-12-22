"Utility objects for pretty syntax in user code."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-12-22"

# Modified by Anders Logg, 2008

from ufl.integral import Integral
from ufl.indexing import Index
from ufl.geometry import Cell

# Default indices
i, j, k, l = [Index() for _i in range(4)]
p, q, r, s = [Index() for _i in range(4)]

# Default integrals
dx = Integral("cell", 0)
ds = Integral("exterior_facet", 0)
dS = Integral("interior_facet", 0)

# Cell types
interval      = Cell("interval", 1)
triangle      = Cell("triangle", 1)
tetrahedron   = Cell("tetrahedron", 1)
quadrilateral = Cell("quadrilateral", 1)
hexahedron    = Cell("hexahedron", 1)
