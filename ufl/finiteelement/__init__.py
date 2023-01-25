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

from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.finiteelement.finiteelement import FiniteElement
from ufl.finiteelement.mixedelement import MixedElement
from ufl.finiteelement.mixedelement import VectorElement
from ufl.finiteelement.mixedelement import TensorElement
from ufl.finiteelement.enrichedelement import EnrichedElement
from ufl.finiteelement.enrichedelement import NodalEnrichedElement
from ufl.finiteelement.restrictedelement import RestrictedElement
from ufl.finiteelement.tensorproductelement import TensorProductElement
from ufl.finiteelement.hdivcurl import HDivElement, HCurlElement, WithMapping
from ufl.finiteelement.brokenelement import BrokenElement

# Export list for ufl.classes
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
    "WithMapping"
    ]
