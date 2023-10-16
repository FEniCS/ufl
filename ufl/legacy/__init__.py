"""Legacy UFL features."""
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

import warnings as _warnings

from ufl.legacy.brokenelement import BrokenElement
from ufl.legacy.enrichedelement import EnrichedElement, NodalEnrichedElement
from ufl.legacy.finiteelement import FiniteElement
from ufl.legacy.finiteelementbase import FiniteElementBase
from ufl.legacy.hdivcurl import HCurlElement, HDivElement, WithMapping
from ufl.legacy.mixedelement import MixedElement, TensorElement, VectorElement
from ufl.legacy.restrictedelement import RestrictedElement
from ufl.legacy.tensorproductelement import TensorProductElement

_warnings.warn("The features in ufl.legacy are deprecated and will be removed in a future version.",
               FutureWarning)
