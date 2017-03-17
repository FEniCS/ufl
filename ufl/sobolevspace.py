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
# Modified by Thomas Gibson 2017

#import six
from ufl.utils.py23 import as_native_str


# @six.python_2_unicode_compatible
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
        self._order = {"L2": 0,
                       "H1": 1,
                       "H2": 2,
                       # Order for the elements below is taken from
                       # its parent Sobolev space
                       "HDiv": 0,
                       "HCurl": 0,
                       "HEin": 0,
                       "HDivDiv": 0,
                       "DirectionalH": 0}[self.name]

    def __unicode__(self):
        # Only in python 2
        return str(self).decode("utf-8")

    def __str__(self):
        return self.name

    def __repr__(self):
        r = "SobolevSpace(%s, %s)" % (repr(self.name), repr(list(self.parents)))
        return as_native_str(r)

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

    def __getitem__(self, spatial_index):
        """Returns the Sobolev space associated with a particular
        spatial coordinate.
        """
        return self

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


class DirectionalSobolevSpace(SobolevSpace):
    """Symbolic representation of a Sobolev space with varying smoothness
    in differerent spatial directions.
    """

    def __init__(self, orders):
        """Instantiate a DirectionalSobolevSpace object.

        :arg orders: an iterable of orders of weak derivatives, where
                     the position denotes in what spatial variable the
                     smoothness requirement is enforced.
        """
        assert all(isinstance(x, int) for x in orders), (
            "Order must be an integer."
        )
        assert all(x < 3 for x in orders), (
            "Not implemented for orders greater than 2"
        )
        name = "DirectionalH"
        parents = [L2]
        super(DirectionalSobolevSpace, self).__init__(name, parents)
        self._orders = tuple(orders)
        self._spatial_indices = range(len(self._orders))

    def __getitem__(self, spatial_index):
        """Returns the Sobolev space associated with a particular
        spatial coordinate.
        """
        if spatial_index not in range(len(self._orders)):
            raise IndexError("Spatial index out of range.")
        spaces = {0: L2,
                  1: H1,
                  2: H2}
        return spaces[self._orders[spatial_index]]

    def __contains__(self, other):
        """Implement `fe in s` where `fe` is a
        :class:`~finiteelement.FiniteElement` and `s` is a
        :class:`DirectionalSobolevSpace`"""
        if isinstance(other, SobolevSpace):
            raise TypeError("Unable to test for inclusion of a " +
                            "SobolevSpace in another SobolevSpace. " +
                            "Did you mean to use <= instead?")
        return (other.sobolev_space() == self or
                all(self[i] in other.sobolev_space().parents
                    for i in self._spatial_indices))

    def __eq__(self, other):
        if isinstance(other, DirectionalSobolevSpace):
            return self._orders == other._orders
        return all(self[i] == other for i in self._spatial_indices)

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper
        subset of."""
        if isinstance(other, DirectionalSobolevSpace):
            if self._spatial_indices != other._spatial_indices:
                return False
            return any(self._orders[i] > other._orders[i]
                       for i in self._spatial_indices)

        if other in [HDiv, HCurl]:
            return all(self._orders[i] >= 1 for i in self._spatial_indices)
        elif other.name in ["HDivDiv", "HEin"]:
            # Don't know how these spaces compare
            return False
        else:
            return any(self._orders[i] > other._order
                       for i in self._spatial_indices)

    def __le__(self, other):
        """In common with intrinsic Python sets, <= indicates "is a subset
        of." """
        if isinstance(other, DirectionalSobolevSpace):
            return (self == other) or (other > self)
        return (self == other) or all(other > self[i]
                                      for i in self._spatial_indices)

    def __gt__(self, other):
        """In common with intrinsic Python sets, > indicates "is a proper
        superset of."""
        if isinstance(other, DirectionalSobolevSpace):
            if self._spatial_indices != other._spatial_indices:
                return False
            return any(self._orders[i] < other._orders[i]
                       for i in self._spatial_indices)

        if other.name in ["HDiv", "HCurl"]:
            return all(self._orders[i] == 0 for i in self._spatial_indices)
        elif other.name in ["HDivDiv", "HEin"]:
            # Don't know how these spaces compare
            return False
        else:
            return any(self._orders[i] < other._order
                       for i in self._spatial_indices)

    def __ge__(self, other):
        """In common with intrinsic Python sets, >= indicates "is a superset
        of." """
        if isinstance(other, DirectionalSobolevSpace):
            return (self == other) or (other < self)
        return (self == other) or all(other < self[i]
                                      for i in self._spatial_indices)

    def __str__(self):
        return self.name + "(%s)" % ", ".join(map(str, self._orders))

    def _repr_latex_(self):
        return "H(%s)" % ", ".join(map(str, self._orders))


L2 = SobolevSpace("L2")
HDiv = SobolevSpace("HDiv", [L2])
HCurl = SobolevSpace("HCurl", [L2])
H1 = SobolevSpace("H1", [HDiv, HCurl, L2])
H2 = SobolevSpace("H2", [H1, HDiv, HCurl, L2])
HEin = SobolevSpace("HEin", [L2])
HDivDiv = SobolevSpace("HDivDiv", [L2])
