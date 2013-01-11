"""Algorithms for working with linearized computational graphs."""

# Copyright (C) 2008-2013 Martin Sandve Alnes
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
# First added:  2008-12-28
# Last changed: 2012-04-12

from collections import defaultdict
from itertools import imap, izip
from heapq import heapify, heappop, heappush

#from ufl import *
from ufl.algorithms.traversal import fast_pre_traversal
from ufl.algorithms.printing import tree_format
from ufl.algorithms.multifunction import MultiFunction
from ufl.classes import Terminal

# O(n) = O(|V|) = O(|E|), since |E| < c|V| for a fairly small c.

#--- Basic utility functions ---

def lists(n):
    return [[] for i in xrange(n)]

def len_items(sequence):
    return map(len, sequence)

def join_lines(sequence):
    return "\n".join(imap(str, sequence))

def all_is(seq1, seq2):
    return all(a is b for (a, b) in izip(seq1, seq2))

def reorder(sequence, ordering):
    "Rearrange the items in a sequence."
    return [sequence[i] for i in ordering]

#--- Graph building functions ---

def build_graph(expr): # O(n)
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
    #for v in post_traversal(expr):
    for v in reversed(list(fast_pre_traversal(expr))):
        i = handled.get(v)
        if i is None:
            i = len(V)
            handled[v] = i
            V.append(v)
            for o in v.operands():
                j = handled[o]
                e = (i, j)
                E.append(e)
    G = V, E
    return G

def extract_incoming_edges(G): # O(n)
    "Build lists of incoming edges to each vertex in a linearized graph."
    V, E = G
    n = len(V)
    Ein = lists(n)
    for i, e in enumerate(E):
        Ein[e[1]].append(i)
    return Ein

def extract_outgoing_edges(G): # O(n)
    "Build list of outgoing edges from each vertex in a linearized graph."
    V, E = G
    n = len(V)
    Eout = lists(n)
    for i, e in enumerate(E):
        Eout[e[0]].append(i)
    return Eout

def extract_edges(G): # O(n)
    """Build lists of incoming and outgoing edges
    to and from each vertex in a linearized graph.

    Returns lists Ein and Eout."""
    V, E = G
    n = len(V)
    Ein  = lists(n)
    Eout = lists(n)
    for i, e in enumerate(E):
        a, b = e
        Ein[b].append(i)
        Eout[a].append(i)
    return Ein, Eout

def extract_incoming_vertex_connections(G): # O(n)
    """Build lists of vertices in incoming and outgoing
    edges to and from each vertex in a linearized graph.

    Returns lists Vin and Vout."""
    V, E = G
    n = len(V)
    Vin  = lists(n)
    for a, b in E:
        Vin[b].append(a)
    return Vin

def extract_outgoing_vertex_connections(G): # O(n)
    """Build lists of vertices in incoming and outgoing
    edges to and from each vertex in a linearized graph.

    Returns lists Vin and Vout."""
    V, E = G
    n = len(V)
    Vout = lists(n)
    for a, b in E:
        Vout[a].append(b)
    return Vout

def extract_vertex_connections(G): # O(n)
    """Build lists of vertices in incoming and outgoing
    edges to and from each vertex in a linearized graph.

    Returns lists Vin and Vout."""
    V, E = G
    n = len(V)
    Vin  = lists(n)
    Vout = lists(n)
    for a, b in E:
        Vin[b].append(a)
        Vout[a].append(b)
    return Vin, Vout

#--- Graph class ---

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

#--- Graph algorithms ---

def format_graph(G):
    V, E = G
    lines = ["Graph with %d vertices and %d edges:" % (len(V), sum(map(len, E)))]
    lines.extend(("", "Vertices:"))
    lines.extend("v%d: \t%s" % (i, v) for (i, v) in enumerate(V))
    lines.extend(("", "Edges:"))
    lines.extend("e%d: \tv%d -> v%d" % (i, e[0], e[1]) for (i, e) in enumerate(E))
    return join_lines(lines)

def find_dependencies(G, targets):
    """Find the set of vertices in a computational
    graph that a set of target vertices depend on."""
    # G is a graph
    V, E = G
    n = len(V)
    # targets is a sequence of vertex indices
    todo = list(targets)
    heapify(todo)

    keep = [False]*n
    while todo:
        t = heappop(todo)
        if not keep[t]:
            keep[t] = True
            for edges in Eout[t]:
                for e in edges:
                    heappush(todo, e[1])
    return keep

#--- Scheduling algorithms ---

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
    q = [HeapItem(Ein_count, Eout_count, i) for i in xrange(n)]
    heapify(q)

    ordering = []
    while q:
        item = heappop(q)
        iv = item.i
        ordering.append(iv)
        for i in Vin[iv]:
            Eout_count[i] -= 1
        # Resort heap, worst case linear time, makes this algorithm O(n^2)... TODO: Not good!
        heapify(q)

    # TODO: Can later accumulate dependencies as well during dft-like algorithm.
    return ordering

#--- Expression tree reconstruction algorithms ---

def rebuild_tree(G):
    """Rebuild expression tree from linearized graph.

    Does not touch the input graph. Assumes the graph
    is directed, acyclic, and connected, such that there
    is only one root node.
    """
    V = G.V()
    E = G.E()
    n = len(V)
    Vout = G.Vout()
    dfo = depth_first_ordering(G)
    subtrees = [None]*n
    for i in dfo:
        v = V[i]
        if not isinstance(v, Terminal):
            # Fetch already reconstructed child vertices
            # and reconstruct non-terminal node from them
            ops = tuple(subtrees[j] for j in Vout[i])
            if all_is(ops, v.operands()):
                pass
            else:
                v = v.reconstruct(*ops)
        subtrees[i] = v
    # Assuming last vertex is the root!
    return v

#--- Graph partitoning ---

class StringDependencyDefiner(MultiFunction):
    """Given an expr, returns a frozenset of its dependencies.

    Possible dependency values are:
        "c"       - depends on runtime information like the cell, local<->global coordinate mappings, facet normals, or coefficients
        "x"       - depends on local coordinates
        "v%d" % i - depends on argument i, for i in [0,rank)
    """
    def __init__(self, argument_deps = None, coefficient_deps = None):
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
        default = frozenset(("v%d" % x.count(), "x"))
        return self.argument_deps.get(x, default)

    def coefficient(self, x):
        default = frozenset(("c", "x"))
        return self.coefficient_deps.get(x, default)

    def constant(self, x):
        default = frozenset(("c",))
        return self.coefficient_deps.get(x, default)

    def geometric_quantity(self, x):
        deps = frozenset(("c", "x",))
        return deps

    def facet_normal(self, o):
        deps = frozenset(("c",))
        # Enabling coordinate dependency for higher order geometries
        # (not handled anywhere else though, so consider this experimental)
        #if o.has_higher_degree_cell_geometry():
        #    deps = deps | frozenset(("x",))
        return deps

    def spatial_derivative(self, o): # TODO: What about (basis) functions of which derivatives are constant? Should special-case spatial_derivative in partition().
        deps = frozenset(("c",))
        # Enabling coordinate dependency for higher order geometries (not handled anywhere else though).
        #if o.has_higher_degree_cell_geometry():
        #    deps = deps | frozenset(("x",))
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
    keys = [None]*n
    for iv, v in enumerate(V):
        # Get keys from all outgoing edges
        edge_keys = [keys[ii] for ii in Vout[iv]]

        # Calculate key for this vertex
        key = criteria(v, edge_keys)

        # Store mappings from key to vertex and back
        partitions[key].append(iv)
        keys[iv] = key
    return partitions, keys

#--- Test code ---

def test_expr():
    from ufl import triangle, FiniteElement, TestFunction, TrialFunction, Coefficient
    element = FiniteElement("CG", triangle, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)
    expr = (f+g)*u.dx(0)*(g-1)*v
    return expr

if __name__ == "__main__":
    expr = test_expr()
    G = Graph(expr)
    V, E = G
    n = len(V)
    Ein = G.Ein()
    Eout = G.Eout()

    print
    print "Entire graph:"
    for iv, v in enumerate(V):
        print "Vertex %03d: %s" % (iv, v)
    for ie, e in enumerate(E):
        print "Edge %03d: %s" % (ie, e)
    for iv, eout in enumerate(Eout):
        print "Edges out for vertex %03d: %s" % (iv, eout)
    for iv, ein in enumerate(Ein):
        print "Edges in for vertex %03d: %s" % (iv, ein)
    print

    from ufl.common import sstr
    partitions, keys = partition(G, string_set_criteria)
    for k in partitions:
        print
        print "Partition with key", sstr(k)
        for iv in partitions[k]:
            print "Vertex %03d: %s" % (iv, V[iv])

if __name__ == "__main_":
    expr = test_expr()

    G = Graph(expr)
    V, E = G
    e2 = rebuild_tree(G)

    Ein  = G.Ein()
    Eout = G.Eout()
    Ein_count  = len_items(Ein)
    Eout_count = len_items(Eout)
    dfo = depth_first_ordering(G)

    print
    print "expr:"
    print expr
    print
    print "e2:"
    print e2
    print
    print tree_format(expr)
    print
    print format_graph(G)
    print
    print "Ein:"
    print join_lines(Ein)
    print
    print "Eout:"
    print join_lines(Eout)
    print
    print "Ein_count:"
    print join_lines(Ein_count)
    print
    print "Eout_count:"
    print join_lines(Eout_count)
    print
    print "dfo:"
    print join_lines(reorder(V, dfo))
