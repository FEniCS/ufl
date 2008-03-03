#!/usr/bin/env python

"""
Polygon attributes.
Maybe add a cell/polygon class for high order mappings like discussed earlier?
"""

valid_polygons = set(("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))

polygon2dim = {
                "interval":       1,
                "triangle":       2,
                "quadrilateral":  2,
                "tetrahedron":    3,
                "hexahedron":     3,
              }

