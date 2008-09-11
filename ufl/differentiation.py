"""Differential operators. Needs work!"""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-26"

from .output import ufl_assert
from .base import UFLObject, Compound
from .indexing import MultiIndex, Index, UnassignedDim, extract_indices
from .variable import Variable
from .tensors import Tensor


# FIXME: This file is not ok! Needs more work!


#--- Differentiation ---

# FIXME: Add SpatialDerivative and TimeDerivative?

class PartialDerivative(UFLObject):
    "Partial derivative of an expression w.r.t. spatial directions given by indices."
    
    __slots__ = ("_expression", "_rank", "_indices", "_free_indices")
    #, "_fixed_indices", "_repeated_indices")
    
    def __init__(self, expression, indices):
        self._expression = expression
        
        if isinstance(indices, MultiIndex): # if constructed from repr
            self._indices = indices
        else:
            self._indices = MultiIndex(indices, len(indices)) # FIXME: Go over the indexing logic here
            # TODO: len(indices)) instead of 1 to support higher order derivatives.
        
        # Find free and repeated indices among the combined
        # indices of the expression and dx((i,j,k))
        indices = expression.free_indices() + self._indices._indices
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) \
            = extract_indices(indices)
        # TODO: We don't need to store all these here, remove the ones we
        #       don't use after implementing summation expansion.
        #self._fixed_indices      = fixed_indices
        self._free_indices       = free_indices
        #self._repeated_indices   = repeated_indices
        self._rank = num_unassigned_indices
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._expression.shape() # FIXME: Wrong with repeated indices.
    
    def __str__(self):
        # TODO: Pretty-print for higher order derivatives.
        return "(d[%s] / dx_%s)" % (self._expression, self._indices)
    
    def __repr__(self):
        return "PartialDerivative(%r, %r)" % (self._expression, self._indices)



# FIXME: Anders: Can't we just remove this?
#        Martin: Not necessarily, not unless PartialDerivative is made more 
#        general. The idea is that Diff represents df/ds where s is a Variable.
# FIXME: This is the same mathematical operation as PartialDerivative, should
#        have very similar behaviour or even be the same class.
class Diff(UFLObject):
    __slots__ = ("_f", "_x", "_index", "_free_indices", "_shape")
    
    def __init__(self, f, x):
        ufl_assert(isinstance(f, UFLObject), "Expecting an UFLObject in Diff.")
        ufl_assert(isinstance(x, Variable), \
            "Expecting a Variable in Diff.")
        self._f = f
        self._x = x
        fi = f.free_indices()
        xi = x.free_indices()
        ufl_assert(len(set(fi) ^ set(xi)) == 0, \
            "Repeated indices in Diff not implemented.") # FIXME
        self._free_indices = tuple(fi + xi)
        self._shape = f.shape() + x.shape() # - repeated_indices FIXME
    
    def operands(self):
        return (self._f, self._x)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._shape
    
    def __str__(self):
        return "(d[%s] / d[%s])" % (self._f, self._x)

    def __repr__(self):
        return "Diff(%r, %r)" % (self._f, self._x)


class Grad(Compound):
    __slots__ = ("_f",)
    
    def __init__(self, f):
        self._f = f
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking gradient of an expression with free indices, should this be a valid expression? Please provide examples!")
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def repeated_indices(self):
        return self._f.repeated_indices()
    
    def shape(self):
        return (UnassignedDim,) + self._f.shape()
    
    def as_basic(self, dim, f):
        ii = Index()
        if f.rank() > 0:
            jj = tuple(Index() for kk in range(f.rank()))
            return Tensor(f[jj].dx(ii), tuple((ii,)+jj))
        else:
            return Tensor(f.dx(ii), (ii,))
    
    def __str__(self):
        return "grad(%s)" % self._f
    
    def __repr__(self):
        return "Grad(%r)" % self._f


class Div(Compound):
    __slots__ = ("_f",)

    def __init__(self, f):
        ufl_assert(f.rank() >= 1, "Can't take the divergence of a scalar.")
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking divergence of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def shape(self):
        return self._f.shape()[1:]
    
    def as_basic(self, dim, f):
        ii = Index()
        if f.rank() == 1:
            g = f[ii]
        else:
            g = f[...,ii]
        return g.dx(ii)

    def __str__(self):
        return "div(%s)" % self._f

    def __repr__(self):
        return "Div(%r)" % self._f


class Curl(Compound):
    __slots__ = ("_f",)

    def __init__(self, f):
        ufl_assert(f.rank()== 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking curl of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def shape(self):
        return (UnassignedDim,)
    
    #def as_basic(self, dim, f):
    #    return FIXME
    
    def __str__(self):
        return "curl(%s)" % self._f
    
    def __repr__(self):
        return "Curl(%r)" % self._f


class Rot(Compound):
    __slots__ = ("_f",)

    def __init__(self, f):
        ufl_assert(f.rank() == 1, "Need a vector.")
        ufl_assert(len(f.free_indices()) == 0, \
            "TODO: Taking rot of an expression with free indices, should this be a valid expression? Please provide examples!")
        self._f = f
    
    def operands(self):
        return (self._f, )
    
    def free_indices(self):
        return self._f.free_indices()
    
    def shape(self):
        return ()
    
    #def as_basic(self, dim, f):
    #    return FIXME
    
    def __str__(self):
        return "rot(%s)" % self._f
    
    def __repr__(self):
        return "Rot(%r)" % self._f

