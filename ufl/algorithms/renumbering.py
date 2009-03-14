__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2009-02-22 -- 2009-03-14"

from ufl.common import Counted
from ufl.indexing import Index, FixedIndex, MultiIndex
from ufl.basisfunction import BasisFunction
from ufl.variable import Label, Variable
from ufl.algorithms.transformations import ReuseTransformer, apply_transformer

class IndexRenumberingTransformer(ReuseTransformer):

    def __init__(self):#, functions, basis_functions):
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
        free_indices = tuple(map(self.index, o.free_indices()))
        return o.reconstruct(free_indices)
    zero = index_annotated
    scalar_value = index_annotated

    def multi_index(self, o):
        return MultiIndex(tuple(map(self.index, o._indices)))
    
    def index(self, o):
        if isinstance(o, FixedIndex):
            return o
        c = o._count
        i = self._index_map.get(c)
        if i is not None:
            return i
        new_index = Index(len(self._index_map))
        self._index_map[c] = new_index
        return new_index

def renumber_indices(expr):
    return apply_transformer(expr, IndexRenumberingTransformer())
