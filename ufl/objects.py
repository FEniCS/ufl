"Utility objects for pretty syntax in user code."

__authors__ = "Martin Sandve Alnes"
__copyright__ = "Copyright (C) 2008-2011 Martin Sandve Alnes"
__license__  = "GNU LGPL version 3 or any later version"
__date__ = "2008-03-14 -- 2011-04-28"

# Modified by Anders Logg, 2008
# Modified by Kristian Oelgaard, 2009

from ufl.indexing import indices
from ufl.integral import Measure
from ufl.geometry import Space, Cell

# Default indices
i, j, k, l = indices(4)
p, q, r, s = indices(4)

# Default measures for integration
dx = Measure(Measure.CELL, 0)
ds = Measure(Measure.EXTERIOR_FACET, 0)
dS = Measure(Measure.INTERIOR_FACET, 0)
dE = Measure(Measure.MACRO_CELL, 0)
dc = Measure(Measure.SURFACE, 0)

# Euclidean spaces
R0 = Space(0)
R1 = Space(1)
R2 = Space(2)
R3 = Space(3)

# Cell types
cell1D        = Cell("cell1D", R1)
cell2D        = Cell("cell2D", R2)
cell3D        = Cell("cell3D", R3)
vertex        = Cell("vertex", R0)
interval      = Cell("interval", R1)
triangle      = Cell("triangle", R2)
tetrahedron   = Cell("tetrahedron", R3)
quadrilateral = Cell("quadrilateral", R2)
hexahedron    = Cell("hexahedron", R3)

# Facet is just a dummy declaration for RestrictedElement
facet = "facet"
