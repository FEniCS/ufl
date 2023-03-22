# -*- coding: utf-8 -*-
# Copyright (C) 2017 Miklós Homolya
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl_legacy.finiteelement.restrictedelement import RestrictedElement
from ufl_legacy.log import deprecate


def InteriorElement(element):
    """Constructs the restriction of a finite element to the interior of
    the cell."""
    deprecate('InteriorElement(element) is deprecated, please use element["interior"] instead.')
    return RestrictedElement(element, restriction_domain="interior")
