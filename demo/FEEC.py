# Copyright (C) 2010 Marie Rognes
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

"""
This demo illustrates the FEEC notation

  V = FiniteElement("P Lambda", cell, r, k)
  V = FiniteElement("P- Lambda", cell, r, k)

and their aliases.
"""
from ufl import (FiniteElement, TestFunction, TestFunctions, TrialFunction,
                 TrialFunctions, dx)
from ufl import exterior_derivative as d
from ufl import inner, interval, tetrahedron, triangle

cells = [interval, triangle, tetrahedron]
r = 1

for cell in cells:
    for family in ["P Lambda", "P- Lambda"]:
        tdim = cell.topological_dimension()
        for k in range(0, tdim + 1):

            # Testing exterior derivative
            V = FiniteElement(family, cell, r, form_degree=k)
            v = TestFunction(V)
            u = TrialFunction(V)

            a = inner(d(u), d(v)) * dx

            # Testing mixed formulation of Hodge Laplace
            if k > 0 and k < tdim + 1:
                S = FiniteElement(family, cell, r, form_degree=k - 1)
                W = S * V
                (sigma, u) = TrialFunctions(W)
                (tau, v) = TestFunctions(W)

                a = (inner(sigma, tau) - inner(d(tau), u) + inner(d(sigma), v) + inner(d(u), d(v))) * dx
