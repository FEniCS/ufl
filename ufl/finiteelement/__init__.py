# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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
# Modified by Andrew T. T. McRae 2014
# Modified by Lawrence Mitchell 2014

from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.finiteelement.finiteelement import FiniteElement
from ufl.finiteelement.mixedelement import MixedElement
from ufl.finiteelement.mixedelement import VectorElement
from ufl.finiteelement.mixedelement import TensorElement
from ufl.finiteelement.enrichedelement import EnrichedElement
from ufl.finiteelement.restrictedelement import RestrictedElement
from ufl.finiteelement.outerproductelement import TensorProductElement
from ufl.finiteelement.outerproductelement import TensorProductVectorElement
from ufl.finiteelement.outerproductelement import TensorProductTensorElement
from ufl.finiteelement.hdivcurl import HDivElement, HCurlElement
from ufl.finiteelement.brokenelement import BrokenElement
from ufl.finiteelement.traceelement import TraceElement
from ufl.finiteelement.facetelement import FacetElement
from ufl.finiteelement.interiorelement import InteriorElement

# Export list for ufl.classes
__all_classes__ = [
    "FiniteElementBase",
    "FiniteElement",
    "MixedElement",
    "VectorElement",
    "TensorElement",
    "EnrichedElement",
    "RestrictedElement",
    "TensorProductElement",
    "OuterProductElement",
    "OuterProductVectorElement",
    "HDivElement",
    "HCurlElement",
    "BrokenElement",
    "TraceElement",
    "FacetElement",
    "InteriorElement",
    ]
