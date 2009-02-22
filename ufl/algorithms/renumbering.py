__authors__ = "Anders Logg"
__date__ = "2009-02-22 -- 2009-02-22"

from ufl.common import Counted
from ufl.indexing import Index, MultiIndex
from ufl.basisfunction import BasisFunction
from ufl.algorithms.transformations import ReuseTransformer, apply_transformer

class RenumberingTransformer(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)
        self._count = 0
        self._map = {}
    
    def expr(self, o, *ops):
        return o.reconstruct(*ops)

    #def terminal(self, o):
    #    if isinstance(o, Counted):
    #        self._renumber(o)
    #    return o

    def multi_index(self, o):
        return MultiIndex(tuple([self.index(index) for index in o]))

    def index(self, o):
        if o in self._map:
            return self._map[o]
        new_index = Index(self._new_count(o))
        self._map[o] = new_index
        return new_index

    def basis_function(self, o):
        return BasisFunction(o.element(), self._new_count(o))

    def _new_count(self, o):
        count = self._count
        self._count += 1
        print "Renumbering: %d --> %d" % (o.count(), count)
        return count

def renumber_indices(expr):
    return apply_transformer(expr, RenumberingTransformer())
