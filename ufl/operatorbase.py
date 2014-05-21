"Base class for all operators, i.e. non-terminal expr types."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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
# Modified by Anders Logg, 2008

from itertools import imap
from ufl.expr import Expr
from ufl.log import error

def _compute_hash(expr): # Best so far
    hashdata = ( (expr.__class__._uflclass,)
            + tuple(hash(o) for o in expr.operands()) )
    return hash(str(hashdata))

_hashes = set()
_hashes_added = 0
def compute_hash_with_stats(expr):
    global _hashes, _hashes_added

    h = compute_hash(expr)

    _hashes.add(h)
    _hashes_added += 1
    if _hashes_added % 10000 == 0:
        print "HASHRATIO", len(_hashes)/float(_hashes_added)

    return h

# This seems to be the best of the above
compute_hash = _compute_hash
#compute_hash = compute_hash_with_stats


#--- Base class for operator objects ---

class Operator(Expr):
    __slots__ = ("_hash",) # TODO: Add _ops tuple here and use that from all operator types instead of separate slots specs for each operator.
    def __init__(self): # *ops):
        Expr.__init__(self)
        self._hash = None
        #self._ops = ops

    def signature_data(self):
        return self._classid

    def __hash__(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        if self._hash is None:
            self._hash = compute_hash(self)
        return self._hash

    def __getnewargs__(self):
        return self.operands()

    def reconstruct(self, *operands):
        "Return a new object of the same type with new operands."
        return self.__class__._uflclass(*operands)

    def is_cellwise_constant(self):
        "Return whether this expression is spatially constant over each cell."
        return all(o.is_cellwise_constant() for o in self.operands())

#--- Subgroups of terminals ---

class AlgebraOperator(Operator):
    __slots__ = ()
    def __init__(self):
        Operator.__init__(self)

class WrapperType(Operator):
    __slots__ = ()
    def __init__(self):
        Operator.__init__(self)

    def shape(self):
        error("A non-tensor type has no shape.")

    def free_indices(self):
        error("A non-tensor type has no indices.")

    def index_dimensions(self):
        error("A non-tensor type has no indices.")
