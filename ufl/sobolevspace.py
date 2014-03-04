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
# First added:  2014-03-04
# Last changed: 2014-03-04


class SobolevSpace(object):
    """Symbolic representation of a Sobolev space. This implements a
    subset of the methods of a Python set so that finite elements and
    other Sobolev spaces can be tested for inclusion.
    """

    def __init__(self, name, parents=None):
        """Instantiate a SobolevSpace object. Name is the name of this space,
parents is a set of Sobolev spaces of which this space is a
subspace."""

        self.name = name
        p = frozenset(parents or [])
        # Ensure that the inclusion operations are transitive.
        self.parents = p.union(*[p_.parents for p_ in p])

    def __str__(self):
        return self.name

    def __repr__(self):
        return "SobolevSpace(%s, %s)" % (self.name, map(str, self.parents))

    def __contains__(self, other):
        """Implement `fe in s` where `fe` is a
        :class:`~finiteelement.FiniteElement` and `s` is a
        :class:`SobolevSpace`"""
        try:
            return (other.sobolev_space is self) \
                or (other.sobolev_space in self.parents)
        except AttributeError:
            if isinstance(other, SobolevSpace):
                raise TypeError("Unable to test for inclusion of a " +
                                "SobolevSpace in another SobolevSpace. " +
                                "Did you mean to use <= instead?")
            else:
                raise

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper
        subset of."""
        return other in self.parents

    def __le__(self, other):
        """In common with intrinsic Python sets, <= indicates "is a subset
        of." """
        return (self is other) or (other in self.parents)

    def __gt__(self, other):
        """In common with intrinsic Python sets, > indicates "is a proper
        subset of."""
        return self in other.parents

    def __ge__(self, other):
        """In common with intrinsic Python sets, >= indicates "is a subset
        of." """
        return (self is other) or (self in other.parents)

L2 = SobolevSpace("L2")
HDiv = SobolevSpace("HDiv", [L2])
HCurl = SobolevSpace("HCurl", [L2])
H1 = SobolevSpace("H1", [HDiv, HCurl, L2])
H2 = SobolevSpace("H2", [H1])
