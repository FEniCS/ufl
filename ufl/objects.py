"""Utility objects for pretty syntax in user code."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-18"

from .base import Number
from .integral import Integral
from .geometry import FacetNormal
from .indexing import Index
#from .finiteelement import Cell

# Default indices
i, j, k, l = [Index() for _i in range(4)]
p, q, r, s = [Index() for _i in range(4)]

# Default integrals
dx0, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9 = [Integral("cell", _domain_id)           for _domain_id in range(10)]
ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7, ds8, ds9 = [Integral("exterior_facet", _domain_id) for _domain_id in range(10)]
dS0, dS1, dS2, dS3, dS4, dS5, dS6, dS7, dS8, dS9 = [Integral("interior_facet", _domain_id) for _domain_id in range(10)]
dx, ds, dS = dx0, ds0, dS0

# Geometric entities
n = FacetNormal()

# Cell types
interval      = "interval"      # Cell("interval", 1)
triangle      = "triangle"      # Cell("triangle", 1)
tetrahedron   = "tetrahedron"   # Cell("tetrahedron", 1)
quadrilateral = "quadrilateral" # Cell("quadrilateral", 1)
hexahedron    = "hexahedron"    # Cell("hexahedron", 1)

