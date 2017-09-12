# -*- coding: utf-8 -*-
# Copyright (C) 2017 Miklós Homolya
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

from ufl.finiteelement.restrictedelement import RestrictedElement
from ufl.log import deprecate


def InteriorElement(element):
    """Constructs the restriction of a finite element to the interior of
    the cell."""
    deprecate('InteriorElement(element) is deprecated, please use element["interior"] instead.')
    return RestrictedElement(element, restriction_domain="interior")
