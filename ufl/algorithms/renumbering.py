__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2009-02-22 -- 2009-03-26"

from ufl.common import Counted
from ufl.log import error
from ufl.expr import Expr
from ufl.indexing import Index, FixedIndex, MultiIndex, Indexed
from ufl.tensors import ComponentTensor
from ufl.basisfunction import BasisFunction
from ufl.variable import Label, Variable
from ufl.algorithms.transformations import ReuseTransformer, apply_transformer

class IndexRenumberingTransformer(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)
        self._index_map = {}
        self._variable_map = {}

    def variable(self, o):
        e, l = o.operands()
        c = l.count()
        v = self._variable_map.get(c)
        if v is None:
            e = self.visit(e)
            l = Label(len(self._variable_map))
            v = Variable(e, l)
            self._variable_map[c] = v
        return v

    def index_annotated(self, o):
        new_indices = tuple(map(self.index, o.free_indices()))
        return o.reconstruct(new_indices)
    zero = index_annotated
    scalar_value = index_annotated

    def multi_index(self, o):
        new_indices = tuple(map(self.index, o._indices))
        return MultiIndex(new_indices)
    
    def index(self, o):
        if isinstance(o, FixedIndex):
            return o
        c = o._count
        i = self._index_map.get(c)
        if i is None:
            i = Index(len(self._index_map))
            self._index_map[c] = i
        return i

    def indexed(self, f):
        g, fi = f.operands()
        if False:# isinstance(g, ComponentTensor):
            h, gi = g.operands()
            if isinstance(h, Indexed):
                # FIXME: This doesn't work when having two levels if this structure, something
                #        like Indexed(ComponentTensor(Indexed(ComponentTensor(Indexed(...)))))
                print "=:"*40
                print "f:"
                print str(f)
                print "fi:"
                print str(fi)
                print 
                print "g:"
                print str(g)
                print "gi:"
                print str(gi)
                print 
                A, hi = h.operands()
                print "h:"
                print str(h)
                print "hi:"
                print str(hi)
                print 
                print "A before:"
                print str(A)
                A = self.visit(A)
                print "A after:"
                print str(A)
                m = dict((i,j) for (i,j) in zip(gi,fi))
                print "m:"
                print str(m)
                #Ai = tuple(self.index(m.get(i,i)) for i in hi)
                Ai = tuple(m.get(i,i) for i in hi)
                print "Ai before:"
                print str(Ai)
                Ai = tuple(self.index(i) for i in Ai)
                print "Ai after:"
                print str(Ai)
                # Note that Ai may contain repeated indices, so don't use []!
                # TODO: If A is a ListTensor, and Ai has fixed indices, try to extract subtensor.
                r = Indexed(A, Ai)
                print "r:"
                print str(r)
                return r
        g  = self.visit(g)
        fi = self.visit(fi)
        r  = self.reuse_if_possible(f, g, fi)
        return r

    def index_sum(self, o, *ops):
        r = self.reuse_if_possible(o, *ops)
        print "=== In index_sum, transformed"
        print "      ", str(o)
        print "  to ", str(r)
        print
        print "operands were:"
        print "\n".join("  " + str(o) for o in o.operands())
        print "operands are now:"
        print "\n".join("  " + str(o) for o in ops)
        print
        return r

    def _component_tensor(self, o, *ops):
        r = self.reuse_if_possible(o, *ops)
        print "=== In component_tensor, transformed"
        print "      ", str(o)
        print "  to ", str(r)
        print
        return r

    def _spatial_derivative(self, o, *ops):
        r = self.reuse_if_possible(o, *ops)
        print "=== In spatial_derivative, transformed"
        print "      ", str(o)
        print "  to ", str(r)
        print
        return r

def renumber_indices(expr):
    if isinstance(expr, Expr) and expr.free_indices():
        error("Not expecting any free indices left in expression.")
    return apply_transformer(expr, IndexRenumberingTransformer())

