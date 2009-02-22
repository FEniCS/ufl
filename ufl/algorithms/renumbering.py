__authors__ = "Anders Logg"
__date__ = "2009-02-22 -- 2009-02-22"

from ufl.common import Counted
from ufl.indexing import MultiIndex
from ufl.algorithms.transformations import ReuseTransformer, apply_transformer

class RenumberingTransformer(ReuseTransformer):

    def __init__(self):
        ReuseTransformer.__init__(self)
        self._count = 0
        self._changed = set()
    
    def expr(self, o, *ops):
        return o.reconstruct(*ops)

    def terminal(self, o):
        if isinstance(o, Counted):
            self._renumber(o)
        return o

    def multi_index(self, o):
        indices = [index for index in o]
        for index in indices:
            self._renumber(index)
        return MultiIndex(tuple(indices))

    def _renumber(self, o):
        if not id(o) in self._changed:
            old_count = o.count()
            new_count = self._count
            o.set_count(new_count)
            self._count += 1
            self._changed.add(id(o))
            print "Renumbering: %d --> %d" % (old_count, new_count)
        return o

def renumber_indices(expr):
    return apply_transformer(expr, RenumberingTransformer())
