"Utility objects for pretty syntax in user code."

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
#
# First added:  2008-03-14
# Last changed: 2011-06-02

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
