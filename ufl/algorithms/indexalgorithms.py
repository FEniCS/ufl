"""This module defines utilities for working
with indices in an expression."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-07 -- 2008-10-30"

from ufl.output import ufl_assert, ufl_error, ufl_warning

# All classes:
from ufl.base import Expr
from ufl.indexing import MultiIndex, Index, Indexed, indices
from ufl.algebra import Product
from ufl.differentiation import SpatialDerivative, VariableDerivative

# Other algorithms:
from ufl.algorithms.transformations import ufl_reuse_handlers, transform

def substitute_indices(expression, indices, values):
    """Substitute Index objects from the list 'indices' with corresponding
    fixed values from the list 'values' in expression."""
    
    ufl_warning("Is this algorithm used? Read comment in source code (indexalgorithms.py).")
    # TODO: Take care when using this, it will replace _all_ occurences of these indices,
    # so in f.ex. (a[i]*b[i]*(1.0 + c[i]*d[i]) the replacement i==0 will result in
    # (a[0]*b[0]*(1.0 + c[0]*d[0]) which is probably not what is wanted.
    # If this is a problem, a new algorithm may be needed.
    
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
        rep_ind = x.repeated_indices()
        return type(x)(*ops) # FIXME 
    d[Product] = e_product
    
    def e_spatial_derivative(x, *ops):
        return x # FIXME
    d[SpatialDerivative] = e_spatial_derivative
    
    def e_variable_derivative(x, *ops):
        return x # FIXME
    d[VariableDerivative] = e_variable_derivative
    
    def e_indexed(x, *ops):
        rep_ind = x.repeated_indices()
        return type(x)(*ops) # FIXME 
    d[Indexed] = e_indexed
    
    return transform(expression, d)

def renumber_indices(expression, offset=0):
    "Given an expression, renumber indices in a contiguous count beginning with offset."
    ufl_assert(isinstance(expression, Expr), "Expecting an Expr.")
    
    ufl_warning("Is this algorithm used? For what reason?")
    
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

