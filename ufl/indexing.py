"""This module defines the single index types and some internal index utilities."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-14 -- 2009-01-28"

from ufl.log import ufl_assert, error
from ufl.common import Counted
from ufl.terminal import Terminal

#--- Index types ---

# TODO: Should we make IndexBase a Terminal? The IndexSum and SpatialDerivative can have an Index instead of a MultiIndex.
#class NewIndexBase(Terminal):
#    def __init__(self):
#        Terminal.__init__(self)
#    
#    def shape(self):
#        error("")
#    
#    def __hash__(self):
#        return hash(repr(self))

class IndexBase(object):
    def __init__(self):
        pass

class Index(IndexBase, Counted):
    __slots__ = ()
    _globalcount = 0
    def __init__(self, count = None):
        IndexBase.__init__(self)
        Counted.__init__(self, count)
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        return isinstance(other, Index) and (self._count == other._count)
    
    def __str__(self):
        c = str(self._count)
        if len(c) > 1:
            c = "{%s}" % c
        return "i_%s" % c
    
    def __repr__(self):
        return "Index(%d)" % self._count

def indices(n):
    "Return a tuple of n new Index objects."
    return tuple(Index() for _i in range(n))

class FixedIndex(IndexBase):
    __slots__ = ("_value",)
    
    def __init__(self, value):
        IndexBase.__init__(self)
        ufl_assert(isinstance(value, int),
            "Expecting integer value for fixed index.")
        self._value = value
    
    def __hash__(self):
        return hash(repr(self))
    
    def __eq__(self, other):
        if isinstance(other, FixedIndex):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == other
        return False
    
    def __str__(self):
        return "%d" % self._value
    
    def __repr__(self):
        return "FixedIndex(%d)" % self._value
 
def as_index_tuple(ii):
    """Takes something the user might input as an index tuple
    inside [], and returns a tuple of actual UFL index objects.
    TODO: Document return value better, in particular 'axes'.
    
    These types are supported:
    - Index
    - int => FixedIndex
    - Complete slice (:) => Axis
    - Ellipsis (...) => multiple Axis
    """
    if isinstance(ii, MultiIndex):
        return ii._indices, ()
    
    if not isinstance(ii, tuple):
        ii = (ii,)
    
    # Convert all indices to Index or FixedIndex objects.
    # If there is an ellipsis, split the indices into before and after.
    pre  = []
    post = []
    axes = set()
    found = False
    for i in ii:
        if i == Ellipsis:
            ufl_assert(not found, "Found duplicate ellipsis.")
            found = True
        else:
            if isinstance(i, int):
                idx = FixedIndex(i)
            elif isinstance(i, (FixedIndex, Index)):
                idx = i
            elif isinstance(i, slice):
                if i == slice(None):
                    idx = Index()
                    axes.add(idx)
                else:
                    # TODO: Use ListTensor for partial slices?
                    error("Partial slices not implemented, only [:]")
            else:
                error("Can't convert this object to index: %r" % i)
            
            if found:
                post.append(idx)
            else:
                pre.append(idx)
    
    # Handle ellipsis as a number of complete slices
    num_axis = len(ii) - len(pre) - len(post)
    ii = indices(num_axis)
    axes.update(ii)
    
    # Construct final tuples to return
    pre.extend(ii)
    pre.extend(post)
    ii = tuple(pre)
    axes = tuple(i for i in ii if i in axes)
    
    return ii, axes

class MultiIndex(Terminal):
    __slots__ = ("_indices",)
    
    def __init__(self, ii):
        Terminal.__init__(self)
        self._indices, axes = as_index_tuple(ii)
        ufl_assert(axes == (), "Not expecting slices at this point.")
    
    def free_indices(self):
        # This reflects the fact that a MultiIndex isn't a tensor expression
        error("Calling free_indices on MultiIndex is an error.")
    
    def index_dimensions(self):
        error("Calling index_dimensions on MultiIndex is an error.")
    
    def shape(self):
        error("Calling shape on MultiIndex is an error.")
    
    def __str__(self):
        return ", ".join(str(i) for i in self._indices)
    
    def __repr__(self):
        return "MultiIndex(%r)" % (self._indices,)
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, i):
        return self._indices[i]
    
    def __iter__(self):
        return iter(self._indices)
    
    def __eq__(self, other):
        return isinstance(other, MultiIndex) and \
            self._indices == other._indices

# TODO: Fix imports everywhere else instead
from ufl.indexutils import complete_shape
from ufl.indexed import Indexed

