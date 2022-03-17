# Copyright (C) 2009 Kristian B. Oelgaard
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
# Restriction of a finite element.
# The below syntax show how one can restrict a higher order Lagrange element
# to only take into account those DOFs that live on the facets.
from ufl import (FiniteElement, TestFunction, TrialFunction, avg, dS, ds,
                 triangle)

# Restricted element
CG_R = FiniteElement("Lagrange", triangle, 4)["facet"]
u_r = TrialFunction(CG_R)
v_r = TestFunction(CG_R)
a = avg(v_r) * avg(u_r) * dS + v_r * u_r * ds
