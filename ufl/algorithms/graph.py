"""Algorithms for working with linearlized computational graphs."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-12-28 -- 2009-01-09"

from itertools import chain, imap, izip
from heapq import heapify, heappop, heappush

from ufl import *
from ufl.algorithms import post_traversal, tree_format
from ufl.classes import Terminal, Variable

def build_graph(expr):
    """Build a linearized graph from an UFL Expr.

    Returns G = (V, E), with V being a list of
    graph nodes (Expr objects) in post traversal
    order and E being a list of edges. Each edge
    is represented as a (i, j) tuple where i and j
    are vertex indices into V.
    """
    V = []
    E = []
    handled = {}
    for v, stack in post_traversal(expr):
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

def extract_incoming_edges(G):
    "Build lists of incoming edges to each vertex in a linearized graph."
    V, E = G
    n = len(V)
    Ein  = [[] for i in xrange(n)]
    for i, e in enumerate(E):
        a, b = e
        Ein[b].append(i)
    return Ein

def extract_outgoing_edges(G):
    "Build list of outgoing edges from each vertex in a linearized graph."
    V, E = G
    n = len(V)
    Eout = [[] for i in xrange(n)]
    for i, e in enumerate(E):
        a, b = e
        Eout[a].append(i)
    return Eout

def extract_edges(G):
    """Build lists of incoming and outgoing edges
    to and from each vertex in a linearized graph.
    
    Returns lists Ein and Eout."""
    V, E = G
    n = len(V)
    Ein  = [[] for i in xrange(n)]
    Eout = [[] for i in xrange(n)]
    for i, e in enumerate(E):
        a, b = e
        Ein[b].append(i)
        Eout[a].append(i)
    return Ein, Eout

def len_items(sequence):
    lens = [len(s) for s in sequence] 
    return lens

def sjoin(sequence):
    return "\n".join(imap(str, sequence))

def print_graph(G):
    V, E = G
    print
    print "V:"
    print "\n".join("v%d: \t%s" % (i, v) for (i, v) in enumerate(V))
    print
    print "E:"
    print "\n".join("e%d: \tv%d -> v%d" % (i, e[0], e[1]) for (i, e) in enumerate(E))
    print

def test_expr():
    cell = triangle
    element = FiniteElement("CG", cell, 1)
    v = TestFunction(element)
    u = TrialFunction(element)
    f = Function(element)
    g = Function(element)
    expr = (f+g)*u*(g-1)*v
    return expr

class HeapItem:
    def __init__(self, ins, outs, i):
        self.ins = ins
        self.outs = outs
        self.i = i

    def __cmp__(self, other):
        a, b = self.outs[self.i], other.outs[other.i]
        if a == b:
            b, a = self.ins[self.i], other.ins[other.i] # FIXME: The other way around?
        return cmp(a, b)

def depth_first_order(G, Ein=None, Eout=None):
    V, E = G
    if Ein is None or Eout is None:
        Ein, Eout = extract_edges(G)
    Ein_count = len_items(Ein)
    Eout_count = len_items(Eout)
    
    # Make a list and a heap of the same items
    q = [HeapItem(Ein_count, Eout_count, i) for i in xrange(len(V))]
    heapify(q)
    
    order = []
    
    while q:
        item = heappop(q)
        iv = item.i
        order.append(iv)
        for ie in Ein[iv]:
            e = E[ie]
            i, j = e
            assert j == iv
            Eout_count[i] -= 1
        # Resort heap, linear time, makes this algorithm O(n^2)...
        heapify(q)
    
    # TODO: Can later accumulate dependencies as well during dft-like algorithm.
    return order

def all_is(seq1, seq2):
    return all(a is b for (a, b) in izip(seq1, seq2))

def rebuild_tree(G):
    """Rebuild expression tree from linearized graph.
    
    Does not touch the input graph. Assumes the graph
    is directed, acyclic, and connected, such that there
    is only one root node.
    """
    V, E = G
    n = len(V)
    Ein, Eout = extract_edges(G)
    dfo = depth_first_order(G, Ein, Eout)
    subtrees = [None]*n
    for i in dfo:
        v = V[i]
        if not isinstance(v, Terminal):
            # Fetch already reconstructed child vertices
            # and reconstruct non-terminal node from them
            ops = tuple(subtrees[E[j][1]] for j in Eout[i])
            if all_is(ops, v.operands()):
                pass
            else:
                v = v._uflclass(*ops)
        subtrees[i] = v
    # Assuming last vertex is the root!
    return v

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

def reorder(sequence, order):
    "Rearrange the items in a sequence."
    return [sequence[i] for i in order]

if __name__ == "__main__":
    
    expr = test_expr()
    
    G = build_graph(expr)
    V, E = G
    e2 = rebuild_tree(G)

    Ein, Eout = extract_edges(G)
    Ein_count = len_items(Ein)
    Eout_count = len_items(Eout)
    dfo = depth_first_order(G)

    print 
    print "expr:"
    print expr
    print 
    print "e2:"
    print e2
    print 
    print tree_format(expr)
    print 
    print_graph(G)
    print 
    print "Ein:"
    print sjoin(Ein)
    print 
    print "Eout:"
    print sjoin(Eout)
    print 
    print "Ein_count:"
    print sjoin(Ein_count)
    print 
    print "Eout_count:"
    print sjoin(Eout_count)
    print 
    print "dfo:"
    print sjoin(reorder(V, dfo))

