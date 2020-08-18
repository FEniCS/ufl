# -*- coding: utf-8 -*-
"""Algorithms for working with linearized computational graphs."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from collections import defaultdict
from heapq import heapify, heappop

from ufl.corealg.traversal import unique_pre_traversal
from ufl.corealg.multifunction import MultiFunction

# O(n) = O(|V|) = O(|E|), since |E| < c|V| for a fairly small c.


# --- Basic utility functions ---

def lists(n):
    return [[] for i in range(n)]


def len_items(sequence):
    return list(map(len, sequence))


# --- Graph building functions ---

def build_graph(expr):  # O(n)
    """Build a linearized graph from an UFL Expr.

    Returns G = (V, E), with V being a list of
    graph nodes (Expr objects) in post traversal
    ordering and E being a list of edges. Each edge
    is represented as a (i, j) tuple where i and j
    are vertex indices into V.
    """
    V = []
    E = []
    handled = {}
    for v in reversed(list(unique_pre_traversal(expr))):
        i = handled.get(v)
        if i is None:
            i = len(V)
            handled[v] = i
            V.append(v)
            for o in v.ufl_operands:
                j = handled[o]
                e = (i, j)
                E.append(e)
    G = V, E
    return G


def extract_incoming_edges(G):  # O(n)
    "Build lists of incoming edges to each vertex in a linearized graph."
    V, E = G
    n = len(V)
    Ein = lists(n)
    for i, e in enumerate(E):
        Ein[e[1]].append(i)
    return Ein


def extract_outgoing_edges(G):  # O(n)
    "Build list of outgoing edges from each vertex in a linearized graph."
    V, E = G
    n = len(V)
    Eout = lists(n)
    for i, e in enumerate(E):
        Eout[e[0]].append(i)
    return Eout


def extract_incoming_vertex_connections(G):  # O(n)
    """Build lists of vertices in incoming and outgoing
    edges to and from each vertex in a linearized graph.

    Returns lists Vin and Vout."""
    V, E = G
    n = len(V)
    Vin = lists(n)
    for a, b in E:
        Vin[b].append(a)
    return Vin


def extract_outgoing_vertex_connections(G):  # O(n)
    """Build lists of vertices in incoming and outgoing
    edges to and from each vertex in a linearized graph.

    Returns lists Vin and Vout."""
    V, E = G
    n = len(V)
    Vout = lists(n)
    for a, b in E:
        Vout[a].append(b)
    return Vout


# --- Graph class ---

class Graph:
    "Graph class which computes connectivity on demand."

    def __init__(self, expression):
        self._V, self._E = build_graph(expression)
        self._Ein = None
        self._Eout = None
        self._Vin = None
        self._Vout = None

    def V(self):
        return self._V

    def E(self):
        return self._E

    def Ein(self):
        if self._Ein is None:
            self._Ein = extract_incoming_edges((self._V, self._E))
        return self._Ein

    def Eout(self):
        if self._Eout is None:
            self._Eout = extract_outgoing_edges((self._V, self._E))
        return self._Eout

    def Vin(self):
        if self._Vin is None:
            self._Vin = extract_incoming_vertex_connections((self._V, self._E))
        return self._Vin

    def Vout(self):
        if self._Vout is None:
            self._Vout = extract_outgoing_vertex_connections((self._V, self._E))
        return self._Vout

    def __iter__(self):
        return iter((self._V, self._E))


# --- Scheduling algorithms ---

class HeapItem(object):
    def __init__(self, incoming, outgoing, i):
        self.incoming = incoming
        self.outgoing = outgoing
        self.i = i

    def __lt__(self, other):
        a = (self.outgoing[self.i], self.incoming[self.i])
        b = (other.outgoing[other.i], other.incoming[other.i])
        return a < b

    def __le__(self, other):
        a = (self.outgoing[self.i], self.incoming[self.i])
        b = (other.outgoing[other.i], other.incoming[other.i])
        return a <= b

    def __eq__(self, other):
        a = (self.outgoing[self.i], self.incoming[self.i])
        b = (other.outgoing[other.i], other.incoming[other.i])
        return a == b


def depth_first_ordering(G):
    V, E = G
    Vin = G.Vin()
    Vout = G.Vout()
    Ein_count = len_items(Vin)
    Eout_count = len_items(Vout)

    # Make a list and a heap of the same items
    n = len(V)
    q = [HeapItem(Ein_count, Eout_count, i) for i in range(n)]
    heapify(q)

    ordering = []
    while q:
        item = heappop(q)
        iv = item.i
        ordering.append(iv)
        for i in Vin[iv]:
            Eout_count[i] -= 1
        # Resort heap, worst case linear time, makes this algorithm
        # O(n^2)... TODO: Not good!
        heapify(q)

    # TODO: Can later accumulate dependencies as well during dft-like
    # algorithm.
    return ordering


# --- Graph partitoning ---

class StringDependencyDefiner(MultiFunction):
    """Given an expr, returns a frozenset of its dependencies.

    Possible dependency values are:
        "c"       - depends on runtime information like the cell, local<->global coordinate mappings, facet normals, or coefficients
        "x"       - depends on local coordinates
        "v%d" % i - depends on argument i, for i in [0,rank)
    """

    def __init__(self, argument_deps=None, coefficient_deps=None):
        MultiFunction.__init__(self)
        if argument_deps is None:
            argument_deps = {}
        if coefficient_deps is None:
            coefficient_deps = {}
        self.argument_deps = argument_deps
        self.coefficient_deps = coefficient_deps

    def expr(self, o):
        return frozenset()

    def argument(self, x):
        default = frozenset(("v%d" % x.number(), "x"))  # TODO: This is missing the part, but this code is ready for deletion anyway?
        return self.argument_deps.get(x, default)

    def coefficient(self, x):
        default = frozenset(("c", "x"))
        return self.coefficient_deps.get(x, default)

    def geometric_quantity(self, x):
        deps = frozenset(("c", "x",))
        return deps

    def facet_normal(self, o):
        deps = frozenset(("c",))
        # Enabling coordinate dependency for higher order geometries
        # (not handled anywhere else though, so consider this experimental)
        # if o.has_higher_degree_cell_geometry():
        #     deps = deps | frozenset(("x",))
        return deps

    def spatial_derivative(self, o):  # TODO: What about (basis) functions of which derivatives are constant? Should special-case spatial_derivative in partition().
        deps = frozenset(("c",))
        # Enabling coordinate dependency for higher order geometries
        # (not handled anywhere else though).
        # if o.has_higher_degree_cell_geometry():
        #     deps = deps | frozenset(("x",))
        return deps


dd = StringDependencyDefiner()


def string_set_criteria(v, keys):
    # Using sets of ufl objects
    key = dd(v)
    for k in keys:
        key |= k
    return frozenset(key)


def partition(G, criteria=string_set_criteria):
    V, E = G
    n = len(V)

    Vout = G.Vout()

    partitions = defaultdict(list)
    keys = [None] * n
    for iv, v in enumerate(V):
        # Get keys from all outgoing edges
        edge_keys = [keys[ii] for ii in Vout[iv]]

        # Calculate key for this vertex
        key = criteria(v, edge_keys)

        # Store mappings from key to vertex and back
        partitions[key].append(iv)
        keys[iv] = key
    return partitions, keys
