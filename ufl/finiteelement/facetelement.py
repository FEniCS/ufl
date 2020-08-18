# -*- coding: utf-8 -*-
# Copyright (C) 2017 Miklós Homolya
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.finiteelement.restrictedelement import RestrictedElement
from ufl.log import deprecate


def FacetElement(element):
    """Constructs the restriction of a finite element to the facets of the
    cell."""
    deprecate('FacetElement(element) is deprecated, please use element["facet"] instead.')
    return RestrictedElement(element, restriction_domain="facet")
