"Base class for all operators, i.e. non-terminal expr types."

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
# Modified by Anders Logg, 2008
#
# First added:  2008-03-14
# Last changed: 2012-05-23

from itertools import imap
from ufl.expr import Expr
from ufl.log import error

# Modified from ufl.algorithms.traveral to avoid circular dependencies...
def traverse_terminals2(expr):
    input = [expr]
    while input:
        e = input.pop()
        ops = e.operands()
        if ops:
            input.extend(ops)
        else:
            yield e

def typetuple(e):
    return tuple(type(o) for o in e.operands())

def compute_hash1(expr): # Crap
    return hash((type(expr), tuple((type(o), typetuple(o)) for o in expr.operands())))

def compute_hash2(expr): # Best so far
    hashdata = ( (expr.__class__._uflclass,)
            + tuple(hash(o) for o in expr.operands()) )
    return hash(str(hashdata))

def compute_hash3(expr): # Crap
    h = (type(expr).__name__,) + expr.operands()
    return hash(h)

def get_some_terminals(expr):
    some = []
    for t in traverse_terminals2(expr):
        some.append(t)
        if len(some) == 5:
            return some
    return some

def compute_hash4(expr): # Not as good as 2
    h = ( (type(expr).__name__,) +
           tuple(type(o).__name__ for o in expr.operands()) +
           tuple(imap(repr, get_some_terminals(expr))) )
    return hash(str(h))

def compute_hash5(expr): # Just as good as 2, with additional work
    hashdata = ((type(expr).__name__,)
             +  tuple(hash(o) for o in expr.operands())
             +  tuple(repr(t) for t in get_some_terminals(expr)))
    return hash(hashdata)

#import md5 # use hashlib instead if we need this
#def compute_hash6(expr): # Exactly as good as 2, with additional work
#    hashdata = ( (expr.__class__._uflclass,)
#            + tuple(hash(o) for o in expr.operands()) )
#    return hash(md5.md5(str(hashdata)).digest())

_hashes = set()
_hashes_added = 0
def compute_hash_with_stats(expr):
    global _hashes, _hashes_added

    h = compute_hash2(expr)
    #h = compute_hash6(expr)

    _hashes.add(h)
    _hashes_added += 1
    if _hashes_added % 10000 == 0:
        print "HASHRATIO", len(_hashes)/float(_hashes_added)

    return h

# This seems to be the best of the above
compute_hash = compute_hash2
#compute_hash = compute_hash_with_stats


#--- Base class for operator objects ---

class Operator(Expr):
    __slots__ = ("_hash",)
    def __init__(self):
        Expr.__init__(self)
        self._hash = None

    def signature_data(self):
        return self._classid

    def __hash__(self):
        "Compute a hash code for this expression. Used by sets and dicts."
        if self._hash is None:
            self._hash = compute_hash(self)
        return self._hash
        #return compute_hash(self) # REPR TODO: Cache or not?

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

#--- Non-tensor types ---

class Tuple(WrapperType):
    "For internal use, never to be created by users."
    __slots__ = ("_items",)
    def __init__(self, *items):
        WrapperType.__init__(self)
        if not all(isinstance(i, Expr) for i in items):
            error("Got non-Expr in Tuple, is this intended? If so, remove this error.")
        self._items = tuple(items)

    def operands(self):
        return self._items

    def shape(self):
        error("Calling shape on a utility type is an error.")

    def free_indices(self):
        error("Calling free_indices on a utility type is an error.")

    def index_dimensions(self):
        error("Calling free_indices on a utility type is an error.")

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __str__(self):
        return "Tuple(*(%s,))" % ", ".join(str(i) for i in self._items)

    def __repr__(self):
        return "Tuple(*%s)" % repr(self._items)

