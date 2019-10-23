# -*- coding: utf-8 -*-
"Utility objects for pretty syntax in user code."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008
# Modified by Kristian Oelgaard, 2009

from ufl.core.multiindex import indices
from ufl.cell import Cell
from ufl.measure import Measure
from ufl.measure import integral_type_to_measure_name

# Default indices
i, j, k, l = indices(4)  # noqa: E741
p, q, r, s = indices(4)

for integral_type, measure_name in integral_type_to_measure_name.items():
    globals()[measure_name] = Measure(integral_type)

# TODO: Firedrake hack, remove later
ds_tb = ds_b + ds_t  # noqa: F821

# Default measure dX including both uncut and cut cells
dX = dx + dC  # noqa: F821

# Create objects for builtin known cell types
vertex = Cell("vertex", 0)
interval = Cell("interval", 1)
triangle = Cell("triangle", 2)
tetrahedron = Cell("tetrahedron", 3)
quadrilateral = Cell("quadrilateral", 2)
hexahedron = Cell("hexahedron", 3)

# Facet is just a dummy declaration for RestrictedElement
facet = "facet"
