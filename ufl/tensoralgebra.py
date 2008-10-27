"""Compound tensor algebra operations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-21"

from .output import ufl_assert
from .base import UFLObject, Terminal
from .indexing import Index, indices, compare_shapes


### Algebraic operations on tensors:
# FloatValues:
#   dot(a,b)      = a*b
#   inner(a,b)    = a*b
#   outer(a,b)    = a*b
# Vectors:
#   dot(u,v)      = u_i v_i
#   inner(u,v)    = u_i v_i
#   outer(u,v)    = A | A_ij = u_i v_j
# Matrices:
#   dot(A,B)      = C | C_ij = A_{ik} B_{kj}
#   inner(A,B)    = A_{ij} B_{ij}
#   outer(A,B)    = C | C_ijkl = A_ij B_kl
# Combined:
#   dot(A,u)      = v | v_i = A_{ik} u_k
#   inner(A,u)    = v | v_i = A_{ik} u_k
#   outer(A,u)    = C | C_ijk = B_ij u_k
#   dot(u,B)      = v | v_i = u_k B_{ki}
#   inner(u,B)    = v | v_i = u_k B_{ki}
#   outer(u,B)    = C | C_ijk = u_i B_jk
#
# Argument requirements:
#   dot(x,y):   last index of x has same dimension as first index of y
#   inner(x,y): shape of x equals the shape of y

class Identity(Terminal):
    __slots__ = ("_dim",)
    
    def __init__(self, dim):
        self._dim = dim
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return (self._dim, self._dim)
    
    def __str__(self):
        return "I"
    
    def __repr__(self):
        return "Identity(%d)" % self._dim
    
    def __eq__(self, other):
        return isinstance(other, Identity) and self._dim == other._dim


# objects representing the operations:

class Transposed(UFLObject):
    __slots__ = ("_A",)
    
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Transposed is only defined for rank 2 tensors.")
        self._A = A
    
    def operands(self):
        return (self._A,)
    
    def free_indices(self):
        return self._A.free_indices()
    
    def shape(self):
        s = self._A.shape()
        return (s[1], s[0])
    
    def __str__(self):
        return "(%s)^T" % self._A
    
    def __repr__(self):
        return "Transposed(%r)" % self._A


class Outer(UFLObject):
    __slots__ = ("_a", "_b", "_free_indices")

    def __init__(self, a, b):
        self._a = a
        self._b = b
        ai = a.free_indices()
        bi = b.free_indices()
        ufl_assert( len(set(ai) ^ set(bi)) == 0, "Didn't expect repeated indices in outer product.") 
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._a.shape() + self._b.shape()
    
    def __str__(self):
        return "(%s) (x) (%s)" % (self._a, self._b)
        #return "%s (x) %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Outer(%r, %r)" % (self._a, self._b)


class Inner(UFLObject):
    __slots__ = ("_a", "_b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() == b.rank(), "Rank mismatch.")
        self._a = a
        self._b = b
        ai = self._a.free_indices()
        bi = self._b.free_indices()
        ufl_assert( len(set(ai) ^ set(bi)) == 0, "Didn't expect repeated indices in outer product.") 
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "(%s) : (%s)" % (self._a, self._b)
        #return "%s : %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Inner(%r, %r)" % (self._a, self._b)


class Dot(UFLObject):
    __slots__ = ("_a", "_b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() >= 1 and b.rank() >= 1,
            "Dot product requires arguments of rank >= 1, got %d and %d." % \
            (a.rank(), b.rank())) # TODO: maybe scalars are ok?
        self._a = a
        self._b = b
        ai = self._a.free_indices()
        bi = self._b.free_indices()
        ufl_assert(len(set(ai) ^ set(bi)) == 0,
                   "Didn't expect repeated indices in outer product.")
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self._a.shape()[:-1] + self._b.shape()[1:]
    
    def __str__(self):
        return "(%s) . (%s)" % (self._a, self._b)
        #return "%s . %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Dot(%r, %r)" % (self._a, self._b)


class Cross(UFLObject):
    __slots__ = ("_a", "_b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() == 1 and b.rank() == 1,
            "Cross product requires arguments of rank 1.")
        self._a = a
        self._b = b
        ai = self._a.free_indices()
        bi = self._b.free_indices()
        ufl_assert( len(set(ai) ^ set(bi)) == 0,
            "Didn't expect repeated indices in outer product.") 
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return (3,)
    
    def __str__(self):
        return "(%s) x (%s)" % (self._a, self._b)
        #return "%s x %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Cross(%r, %r)" % (self._a, self._b)


class Trace(UFLObject):
    __slots__ = ("_A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Trace of tensor with rank != 2 is undefined.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return self._A.free_indices()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "tr(%s)" % self._A
    
    def __repr__(self):
        return "Trace(%r)" % self._A


class Determinant(UFLObject):
    __slots__ = ("_A",)

    def __init__(self, A):
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 0 or r == 2,
            "Determinant of tensor with rank != 2 is undefined.")
        ufl_assert(r == 0 or compare_shapes((sh[0],), (sh[1],)),
            "Cannot take determinant of rectangular rank 2 tensor.")
        ufl_assert(len(A.free_indices()) == 0,
            "Didn't expect free indices in determinant.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "det(%s)" % self._A
    
    def __repr__(self):
        return "Determinant(%r)" % self._A


class Inverse(UFLObject): # TODO: Drop Inverse and represent it as product of Determinant and Cofactor?
    __slots__ = ("_A",)

    def __init__(self, A):
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 0 or r == 2, "Inverse of tensor with rank != 2 or 0 is undefined.")
        ufl_assert(r == 0 or compare_shapes((sh[0],), (sh[1],)),
            "Cannot take inverse of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Inverse.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._A.shape()
    
    def __str__(self):
        return "(%s)^-1" % self._A
    
    def __repr__(self):
        return "Inverse(%r)" % self._A


class Cofactor(UFLObject):
    __slots__ = ("_A",)

    def __init__(self, A):
        sh = A.shape()
        ufl_assert(len(sh) == 2, "Cofactor of tensor with rank != 2 is undefined.")
        ufl_assert(sh[0] == sh[1], "Cannot take cofactor of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Cofactor.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._A.shape()

    def __str__(self):
        return "cofactor(%s)" % self._A
    
    def __repr__(self):
        return "Cofactor(%r)" % self._A


class Deviatoric(UFLObject):
    __slots__ = ("_A",)

    def __init__(self, A):
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 2, "Deviatoric part of tensor with rank != 2 is undefined.")
        ufl_assert(compare_shapes((sh[0],), (sh[1],)),
            "Cannot take deviatoric part of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Deviatoric.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return self._A.free_indices()
    
    def shape(self):
        return self._A.shape()
    
    def __str__(self):
        return "dev(%s)" % self._A
    
    def __repr__(self):
        return "Deviatoric(%r)" % self._A


class Skew(UFLObject):
    __slots__ = ("_A",)

    def __init__(self, A):
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 2, "Skew part of tensor with rank != 2 is undefined.")
        ufl_assert(compare_shapes((sh[0],), (sh[1],)),
            "Cannot take skew part of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Skew.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return self._A.free_indices()
    
    def shape(self):
        return self._A.shape()
    
    def __str__(self):
        return "skew(%s)" % self._A
    
    def __repr__(self):
        return "Skew(%r)" % self._A
