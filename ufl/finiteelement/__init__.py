"This module defines the UFL finite element classes."

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
#
# First added:  2008-03-03
# Last changed: 2012-08-16

from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.finiteelement.finiteelement import FiniteElement
from ufl.finiteelement.mixedelement import MixedElement
from ufl.finiteelement.mixedelement import VectorElement
from ufl.finiteelement.mixedelement import TensorElement
from ufl.finiteelement.enrichedelement import EnrichedElement
from ufl.finiteelement.restrictedelement import RestrictedElement
from ufl.finiteelement.tensorproductelement import TensorProductElement
