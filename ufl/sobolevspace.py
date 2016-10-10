# -*- coding: utf-8 -*-
"""This module defines a symbolic heirarchy of Sobolev spaces to enable
symbolic reasoning about the spaces in which finite elements lie."""

# Copyright (C) 2014 Imperial College London and others
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
# Written by David Ham 2014
#
# Modified by Martin Alnaes 2014
# Modified by Lizao Li 2015


class SobolevSpace(object):
    """Symbolic representation of a Sobolev space. This implements a
    subset of the methods of a Python set so that finite elements and
    other Sobolev spaces can be tested for inclusion.
    """

    def __init__(self, name, parents=None):
        """Instantiate a SobolevSpace object.

        :param name: The name of this space,
        :param parents: A set of Sobolev spaces of which this
        space is a subspace."""

        self.name = name
        p = frozenset(parents or [])
        # Ensure that the inclusion operations are transitive.
        self.parents = p.union(*[p_.parents for p_ in p])

    def __unicode__(self):
        # Only in python 2
        return str(self).decode("utf-8")

    def __bytes__(self):
        # Only in python 3
        return str(self).encode("utf-8")

    def __str__(self):
        return self.name

    def __repr__(self):
        return "SobolevSpace(%r, %r)" % (self.name, list(self.parents))

    def _repr_latex_(self):
        if len(self.name) == 2:
            return "$%s^%s$" % tuple(self.name)
        else:
            return "$%s(%s)$" % (self.name[0], self.name[1:].lower())

    def __eq__(self, other):
        return isinstance(other, SobolevSpace) and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(("SobolevSpace", self.name))

    def __contains__(self, other):
        """Implement `fe in s` where `fe` is a
        :class:`~finiteelement.FiniteElement` and `s` is a
        :class:`SobolevSpace`"""
        if isinstance(other, SobolevSpace):
            raise TypeError("Unable to test for inclusion of a " +
                            "SobolevSpace in another SobolevSpace. " +
                            "Did you mean to use <= instead?")
        return (other.sobolev_space() == self or
                self in other.sobolev_space().parents)

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper
        subset of."""
        return other in self.parents

    def __le__(self, other):
        """In common with intrinsic Python sets, <= indicates "is a subset
        of." """
        return (self == other) or (other in self.parents)

    def __gt__(self, other):
        """In common with intrinsic Python sets, > indicates "is a proper
        superset of."""
        return self in other.parents

    def __ge__(self, other):
        """In common with intrinsic Python sets, >= indicates "is a superset
        of." """
        return (self == other) or (self in other.parents)

    def __call__(self, element):
        """Syntax shortcut to create a HDivElement or HCurlElement."""
        if self.name == "HDiv":
            from ufl.finiteelement import HDivElement
            return HDivElement(element)
        elif self.name == "HCurl":
            from ufl.finiteelement import HCurlElement
            return HCurlElement(element)
        raise NotImplementedError("SobolevSpace has no call operator (only the specific HDiv and HCurl instances).")

L2 = SobolevSpace("L2")
HDiv = SobolevSpace("HDiv", [L2])
HCurl = SobolevSpace("HCurl", [L2])
H1 = SobolevSpace("H1", [HDiv, HCurl, L2])
H2 = SobolevSpace("H2", [H1])
HEin = SobolevSpace("HEin", [L2])
HDivDiv = SobolevSpace("HDivDiv", [L2])
