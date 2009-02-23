__authors__ = "Anders Logg"
__date__ = "2009-02-22 -- 2009-02-23"

from ufl.common import Counted
from ufl.indexing import Index, MultiIndex
from ufl.basisfunction import BasisFunction
from ufl.algorithms.transformations import ReuseTransformer, apply_transformer

class IndexRenumberingTransformer(ReuseTransformer):

    def __init__(self):#, functions, basis_functions):
        ReuseTransformer.__init__(self)
        self._index_map = {}

    def multi_index(self, o):
        return MultiIndex(tuple(self.index(index) for index in o))

    def index(self, o):
        if isinstance(o, FixedIndex):
            return o
        i = self._index_map.get(o)
        if i is not None:
            return i
        new_count = len(self._index_map)
        new_index = Index(new_count)
        print "Renumbering: %d --> %d" % (o.count(), new_count)
        self._index_map[o] = new_index
        return new_index

def renumber_indices(expr):
    return apply_transformer(expr, IndexRenumberingTransformer())
