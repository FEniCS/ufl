"Utility objects for pretty syntax in user code."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-11-05"

# Modified by Anders Logg, 2008

from .integral import Integral
from .indexing import Index

# Default indices
i, j, k, l = [Index() for _i in range(4)]
p, q, r, s = [Index() for _i in range(4)]

# Default integrals
dx = Integral("cell", 0)
ds = Integral("exterior_facet", 0)
dS = Integral("interior_facet", 0)

# Cell types
interval      = "interval"      # Cell("interval", 1)
triangle      = "triangle"      # Cell("triangle", 1)
tetrahedron   = "tetrahedron"   # Cell("tetrahedron", 1)
quadrilateral = "quadrilateral" # Cell("quadrilateral", 1)
hexahedron    = "hexahedron"    # Cell("hexahedron", 1)
