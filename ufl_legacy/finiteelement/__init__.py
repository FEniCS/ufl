# -*- coding: utf-8 -*-
# flake8: noqa
"This module defines the UFL finite element classes."

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Andrew T. T. McRae 2014
# Modified by Lawrence Mitchell 2014

from ufl_legacy.finiteelement.finiteelementbase import FiniteElementBase
from ufl_legacy.finiteelement.finiteelement import FiniteElement
from ufl_legacy.finiteelement.mixedelement import MixedElement
from ufl_legacy.finiteelement.mixedelement import VectorElement
from ufl_legacy.finiteelement.mixedelement import TensorElement
from ufl_legacy.finiteelement.enrichedelement import EnrichedElement
from ufl_legacy.finiteelement.enrichedelement import NodalEnrichedElement
from ufl_legacy.finiteelement.restrictedelement import RestrictedElement
from ufl_legacy.finiteelement.tensorproductelement import TensorProductElement
from ufl_legacy.finiteelement.hdivcurl import HDivElement, HCurlElement, WithMapping
from ufl_legacy.finiteelement.brokenelement import BrokenElement
from ufl_legacy.finiteelement.facetelement import FacetElement
from ufl_legacy.finiteelement.interiorelement import InteriorElement

# Export list for ufl_legacy.classes
__all_classes__ = [
    "FiniteElementBase",
    "FiniteElement",
    "MixedElement",
    "VectorElement",
    "TensorElement",
    "EnrichedElement",
    "NodalEnrichedElement",
    "RestrictedElement",
    "TensorProductElement",
    "HDivElement",
    "HCurlElement",
    "BrokenElement",
    "FacetElement",
    "InteriorElement",
    "WithMapping"
    ]
