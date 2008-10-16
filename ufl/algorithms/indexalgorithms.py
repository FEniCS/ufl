"""This module defines utilities for working
with indices in an expression."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-02"

from ..output import ufl_assert, ufl_error

# All classes:
from ..base import UFLObject
from ..indexing import MultiIndex, Indexed, Index, FixedIndex
#from ..indexing import AxisType, as_index, as_index_tuple, extract_indices
from ..tensors import ListTensor, ComponentTensor
from ..algebra import Product
from ..differentiation import SpatialDerivative, Diff

# Lists of all UFLObject classes
from ..classes import ufl_classes, terminal_classes, nonterminal_classes, compound_classes

# Other algorithms:
from .analysis import basisfunctions, coefficients, indices, duplications
from .transformations import ufl_reuse_handlers, transform


# TODO: Take care when using this, it will replace _all_ occurences of these indices,
# so in f.ex. (a[i]*b[i]*(1.0 + c[i]*d[i]) the replacement i==0 will result in
# (a[0]*b[0]*(1.0 + c[0]*d[0]) which is probably not what is wanted.
# If this is a problem, a new algorithm may be needed.
def substitute_indices(expression, indices, values):
    """Substitute Index objects from the list 'indices' with corresponding
    fixed values from the list 'values' in expression."""
    d = ufl_reuse_handlers()

    def s_multi_index(x, *ops):
        newindices = []
        for i in x:
            try:
                idx = indices.index(i)
                val = values[idx]
                newindices.append(val)
            except:
                newindices.append(i)
        return MultiIndex(*newindices)
    d[MultiIndex] = s_multi_index

    return transform(expression, d)


def expand_indices(expression):
    "Expand implicit summations into explicit Sums of Products."
    ufl_error("Not implemented.")
    
    d = ufl_reuse_handlers()
    
    def e_product(x, *ops):
        rep_ind = x._repeated_indices
        return x.__class__(*ops) # FIXME 
    d[Product] = e_product
    
    def e_partial_diff(x, *ops):
        return x # FIXME
    d[PartialDiff] = e_partial_diff
    
    def e_diff(x, *ops):
        return x # FIXME
    d[Diff] = e_diff
    
    return transform(expression, d)


def renumber_indices(expression, offset=0):
    "Given an expression, renumber indices in a contiguous count beginning with offset."
    ufl_assert(isinstance(expression, UFLObject), "Expecting an UFLObject.")
    
    # Build a set of all indices used in expression
    idx = indices(expression)
    
    # Build an index renumbering mapping
    k = offset
    indexmap = {}
    for i in idx:
        if i not in indexmap:
            indexmap[i] = Index(count=k)
            k += 1
    
    # Apply index mapping
    handlers = ufl_reuse_handlers()
    def multi_index_handler(o):
        ind = []
        for i in o:
            if isinstance(i, Index):
                ind.append(indexmap[i])
            else:
                ind.append(i)
        return MultiIndex(tuple(ind), len(o))
    handlers[MultiIndex] = multi_index_handler
    return transform(expression, handlers)

