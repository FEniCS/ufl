"""Algorithms for working with linearized computational graphs."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-12-28 -- 2009-02-26"

from collections import defaultdict
from itertools import chain, imap, izip
from heapq import heapify, heappop, heappush

from ufl import *
from ufl.algorithms.traversal import post_traversal
from ufl.algorithms.printing import tree_format
from ufl.algorithms.transformations import MultiFunction
from ufl.classes import Terminal, Variable

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
    #for v in reversed(fast_pre_traversal(expr)):
    for v in post_traversal(expr):
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
        a, b = e
        Ein[b].append(i)
    return Ein

def extract_outgoing_edges(G): # O(n)
    "Build list of outgoing edges from each vertex in a linearized graph."
    V, E = G
    n = len(V)
    Eout = lists(n)
    for i, e in enumerate(E):
        a, b = e
        Eout[a].append(i)
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

    def __cmp__(self, other):
        a = self.outgoing[self.i]
        b = other.outgoing[other.i]
        if a == b:
            a = self.incoming[self.i]   # FIXME: The other way around?
            b = other.incoming[other.i] # FIXME: The other way around?
        return cmp(a, b)

def depth_first_ordering(G):
    V, E = G
    Ein = G.Ein() # TODO: Can use extract_vertex_connections
    Eout = G.Eout()
    Ein_count = len_items(Ein)
    Eout_count = len_items(Eout)
    
    # Make a list and a heap of the same items
    n = len(V)
    q = [HeapItem(Ein_count, Eout_count, i) for i in xrange(n)]
    heapify(q)
    
    ordering = []
    while q:
        item = heappop(q)
        iv = item.i
        ordering.append(iv)
        #for i in Vin[iv]: # TODO: Faster to use this
        #    Eout_count[i] -= 1
        for ie in Ein[iv]:
            e = E[ie]
            i, j = e
            assert j == iv
            Eout_count[i] -= 1
        # Resort heap, worst case linear time, makes this algorithm O(n^2)... FIXME: Not good!
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
    Eout = G.Eout()# TODO: Can use extract_vertex_connections
    dfo = depth_first_ordering(G)
    subtrees = [None]*n
    for i in dfo:
        v = V[i]
        if not isinstance(v, Terminal):
            # Fetch already reconstructed child vertices
            # and reconstruct non-terminal node from them
            ops = tuple(subtrees[E[j][1]] for j in Eout[i])
            #ops = tuple(subtrees[j] for j in Vout[i]) # TODO: Use this instead (is it Vout or Vin?)
            if all_is(ops, v.operands()):
                pass
            else:
                v = v.reconstruct(*ops)
        subtrees[i] = v
    # Assuming last vertex is the root!
    return v

#--- Graph partitoning ---

class StringDependencyDefiner(MultiFunction):
    """Returns a set of direct dependencies (as strings) given an expr.
    
    Possible dependency values are:
        "c"  - depends on cell
        "n"  - depends on facet
        "v%d" % i - depends on basis function i
        "w"  - depends on coefficients
        "xi" - depends on local coordinates
        "x"  - depends on global coordinates
    """
    def __init__(self, basis_function_deps = None, function_deps = None):
        MultiFunction.__init__(self)
        if basis_function_deps is None:
            basis_function_deps = {}
        if function_deps is None:
            function_deps = {}
        self.basis_function_deps = basis_function_deps
        self.function_deps = function_deps
    
    def expr(self, o):
        return set()
    
    def basis_function(self, x):
        default = set(("v%d" % x.count(),))
        return self.basis_function_deps.get(x, default)
    
    def function(self, x):
        default = set(("w", "c", "x"))
        return self.function_deps.get(x, default)
    
    def constant(self, x):
        default = set(("w", "c"))
        return self.function_deps.get(x, default)
    
    def facet_normal(self, o):
        deps = set(("c",))
        # Enabling coordinate dependency for higher order geometries (not handled anywhere else though).
        if o.cell().degree() > 1:
            deps.add("x")
        return deps
    
    def spatial_derivative(self, o): # FIXME: What about (basis) functions of which derivatives are constant? Should special-case spatial_derivative in partition().
        deps = set(("c",))
        # Enabling coordinate dependency for higher order geometries (not handled anywhere else though).
        if o.cell().degree() > 1:
            deps.add("x")
        return deps

dd = StringDependencyDefiner()

def string_set_criteria(v, keys):
    # Using sets of ufl objects
    key = dd(v)
    for k in keys:
        key |= k
    return frozenset(key)

def partition(G, criteria):
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
    cell = triangle
    element = FiniteElement("CG", cell, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    f = Function(element)
    g = Function(element)
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
#if __name__ == "__main__":
    
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

