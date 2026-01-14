"""Sobolev spaces.

This module defines a symbolic heirarchy of Sobolev spaces to enable
symbolic reasoning about the spaces in which finite elements lie.
"""
# Copyright (C) 2014 Imperial College London and others
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David Ham 2014
#
# Modified by Martin Alnaes 2014
# Modified by Lizao Li 2015
# Modified by Thomas Gibson 2017

from functools import total_ordering
from math import inf, isinf

__all_classes__ = ["SobolevSpace", "DirectionalSobolevSpace"]


@total_ordering
class SobolevSpace:
    """Symbolic representation of a Sobolev space.

    This implements a subset of the methods of a Python set so that
    finite elements and other Sobolev spaces can be tested for
    inclusion.
    """

    def __init__(self, name, parents=None):
        """Instantiate a SobolevSpace object.

        Args:
            name: The name of this space,
            parents: A set of Sobolev spaces of which this
            space is a subspace.
        """
        self.name = name
        p = frozenset(parents or [])
        # Ensure that the inclusion operations are transitive.
        self.parents = p.union(*[p_.parents for p_ in p])
        order_dict = {"L2": 0, "H1": 1, "H2": 2, "H3": 3, "HInf": inf}
        try:
            order = order_dict[self.name]
        except KeyError:
            # Take the maximum order over all the parents
            # For instance, H1Div has order=1 because it has H1 as parent
            if len(self.parents) == 0:
                order = 0
            else:
                order = max(p._order for p in self.parents)
        self._order = order

    def __str__(self):
        """Format as a string."""
        return self.name

    def __repr__(self):
        """Representation."""
        return f"SobolevSpace({self.name!r}, {list(self.parents)!r})"

    def __eq__(self, other):
        """Check equality."""
        return isinstance(other, SobolevSpace) and self.name == other.name

    def __ne__(self, other):
        """Not equal."""
        return not self == other

    def __hash__(self):
        """Hash."""
        return hash(("SobolevSpace", self.name))

    def __getitem__(self, spatial_index):
        """Returns the Sobolev space associated with a particular spatial coordinate."""
        return self

    def __contains__(self, other):
        """Implement `fe in s` where `fe` is a FiniteElement and `s` is a SobolevSpace."""
        if isinstance(other, SobolevSpace):
            raise TypeError(
                "Unable to test for inclusion of a SobolevSpace in another SobolevSpace. "
                "Did you mean to use <= instead?"
            )
        return other.sobolev_space == self or self in other.sobolev_space.parents

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper subset of"."""
        return other in self.parents


@total_ordering
class DirectionalSobolevSpace(SobolevSpace):
    """Directional Sobolev space.

    Symbolic representation of a Sobolev space with varying smoothness
    in different spatial directions.
    """

    def __init__(self, orders):
        """Instantiate a DirectionalSobolevSpace object.

        Args:
            orders: an iterable of orders of weak derivatives, where
                the position denotes in what spatial variable the
                smoothness requirement is enforced.
        """
        assert all(isinstance(x, int) or isinf(x) for x in orders), (
            "Order must be an integer or infinity."
        )
        name = "DirectionalH"
        parents = [L2]
        super().__init__(name, parents)
        self._orders = tuple(orders)
        self._spatial_indices = range(len(self._orders))

    def __getitem__(self, spatial_index):
        """Returns the Sobolev space associated with a particular spatial coordinate."""
        if spatial_index not in range(len(self._orders)):
            raise IndexError("Spatial index out of range.")
        spaces = {0: L2, 1: H1, 2: H2, 3: H3, inf: HInf}
        return spaces[self._orders[spatial_index]]

    def __contains__(self, other):
        """Check if one space is contained in another.

        Implement `fe in s` where `fe` is a FiniteElement and `s` is a
        DirectionalSobolevSpace.
        """
        if isinstance(other, SobolevSpace):
            raise TypeError(
                "Unable to test for inclusion of a SobolevSpace in another SobolevSpace. "
                "Did you mean to use <= instead?"
            )
        return other.sobolev_space == self or all(
            self[i] in other.sobolev_space.parents for i in self._spatial_indices
        )

    def __eq__(self, other):
        """Check equality."""
        if isinstance(other, DirectionalSobolevSpace):
            return self._orders == other._orders
        return all(self[i] == other for i in self._spatial_indices)

    def __lt__(self, other):
        """In common with intrinsic Python sets, < indicates "is a proper subset of."""
        if isinstance(other, DirectionalSobolevSpace):
            if self._spatial_indices != other._spatial_indices:
                return False
            return any(self._orders[i] > other._orders[i] for i in self._spatial_indices)

        if other in [HDiv, HCurl]:
            return all(self._orders[i] >= 1 for i in self._spatial_indices)
        elif other.name in ["HDivDiv", "HEin", "HCurlDiv"]:
            # Don't know how these spaces compare
            return NotImplementedError(f"Don't know how to compare with {other.name}")
        else:
            return any(self._orders[i] > other._order for i in self._spatial_indices)

    def __str__(self):
        """Format as a string."""
        return f"{self.name}({', '.join(map(str, self._orders))})"


L2 = SobolevSpace("L2")
HDiv = SobolevSpace("HDiv", [L2])
HCurl = SobolevSpace("HCurl", [L2])
H1 = SobolevSpace("H1", [HCurl, HDiv, L2])
H1Div = SobolevSpace("H1Div", [H1])
H1Curl = SobolevSpace("H1Curl", [H1])
H2 = SobolevSpace("H2", [H1Curl, H1Div, H1])
H3 = SobolevSpace("H3", [H2])
HInf = SobolevSpace("HInf", [H3])
HEin = SobolevSpace("HEin", [L2])
HDivDiv = SobolevSpace("HDivDiv", [L2])
HCurlDiv = SobolevSpace("HCurlDiv", [L2])
