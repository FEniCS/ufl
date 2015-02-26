"Utility objects for pretty syntax in user code."

# Copyright (C) 2008-2014 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008
# Modified by Kristian Oelgaard, 2009

from ufl.core.multiindex import indices
from ufl.integral import Measure
from ufl.geometry import Cell

# Default indices
i, j, k, l = indices(4)
p, q, r, s = indices(4)

# Default measures for integration
dx = Measure("cell")
ds = Measure("exterior_facet")
ds_b = Measure("exterior_facet_bottom")
ds_t = Measure("exterior_facet_top")
ds_v = Measure("exterior_facet_vert")
ds_tb = ds_b + ds_t
dS = Measure("interior_facet")
dS_h = Measure("interior_facet_horiz")
dS_v = Measure("interior_facet_vert")
dP = Measure("vertex")
dc = Measure("custom")
#dE = Measure("macro_cell")

# Cell types
cell2D        = Cell("cell2D", 2)
cell3D        = Cell("cell3D", 3)
vertex        = Cell("vertex", 0)
interval      = Cell("interval", 1)
triangle      = Cell("triangle", 2)
tetrahedron   = Cell("tetrahedron", 3)
quadrilateral = Cell("quadrilateral", 2)
hexahedron    = Cell("hexahedron", 3)

# Facet is just a dummy declaration for RestrictedElement
facet = "facet"
