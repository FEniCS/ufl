"""Compound tensor algebra operations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-18"

from .output import ufl_assert
from .base import UFLObject, Terminal, Compound


### Algebraic operations on tensors:
# Scalars:
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
    __slots__ = ()
    
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


# objects representing the operations:

class Transposed(Compound):
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
    
    def as_basic(self, A):
        ii = Index()
        jj = Index()
        return Tensor(A[ii, jj], (jj, ii))
    
    def __str__(self):
        return "(%s)^T" % self._A
    
    def __repr__(self):
        return "Transposed(%r)" % self._A


class Outer(Compound):
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

    def as_basic(self, a, b):
        ii = tuple(Index() for kk in range(a.rank()))
        jj = tuple(Index() for kk in range(b.rank()))
        return a[ii]*b[jj]
    
    def __str__(self):
        return "(%s) (x) (%s)" % (self._a, self._b)
        #return "%s (x) %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Outer(%r, %r)" % (self._a, self._b)


class Inner(Compound):
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
    
    def as_basic(self, a, b):
        ii = tuple(Index() for jj in range(a.rank()))
        return a[ii]*b[ii]
    
    def __str__(self):
        return "(%s) : (%s)" % (self._a, self._b)
        #return "%s : %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Inner(%r, %r)" % (self._a, self._b)


class Dot(Compound):
    __slots__ = ("_a", "_b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() >= 1 and b.rank() >= 1, "Dot product requires arguments of rank >= 1, got %d and %d." % (a.rank(), b.rank())) # TODO: maybe scalars are ok?
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
        return self._a.shape()[:-1] + self._b.shape()[1:]
    
    def as_basic(self, a, b):
        ii = Index()
        aa = a[ii] if (a.rank() == 1) else a[...,ii]
        bb = b[ii] if (b.rank() == 1) else b[ii,...]
        return aa*bb
    
    def __str__(self):
        return "(%s) . (%s)" % (self._a, self._b)
        #return "%s . %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Dot(%r, %r)" % (self._a, self._b)


class Cross(Compound):
    __slots__ = ("_a", "_b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() == 1 and b.rank() == 1, "Cross product requires arguments of rank 1.")
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
        return (3,)
    
    #def as_basic(self, a, b):
    #    return FIXME
    
    def __str__(self):
        return "(%s) x (%s)" % (self._a, self._b)
        #return "%s x %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Cross(%r, %r)" % (self._a, self._b)


class Trace(Compound):
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
    
    def as_basic(self, A):
        i = Index()
        return A[i,i]
    
    def __str__(self):
        return "tr(%s)" % self._A
    
    def __repr__(self):
        return "Trace(%r)" % self._A


class Determinant(Compound):
    __slots__ = ("_A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Determinant of tensor with rank != 2 is undefined.")
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in determinant.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return ()
    
    #def as_basic(self, A):
    #    return FIXME
    
    def __str__(self):
        return "det(%s)" % self._A
    
    def __repr__(self):
        return "Determinant(%r)" % self._A


class Inverse(Compound):
    __slots__ = ("_A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Inverse of tensor with rank != 2 is undefined.")
        s = A.shape()
        ufl_assert(s[0] == s[1], "Cannot take inverse of rectangular matrix with dimensions %s." % repr(s))
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Inverse.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return A.shape()
    
    #def as_basic(self, A):
    #    return FIXME
    
    def __str__(self):
        return "(%s)^-1" % self._A
    
    def __repr__(self):
        return "Inverse(%r)" % self._A


class Deviatoric(Compound):
    __slots__ = ("_A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Deviatoric part of tensor with rank != 2 is undefined.")
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Deviatoric.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return self._A.free_indices()
    
    def shape(self):
        return self._A.shape()
    
    #def as_basic(self, A):
    #    return FIXME
    
    def __str__(self):
        return "dev(%s)" % self._A
    
    def __repr__(self):
        return "Deviatoric(%r)" % self._A


class Cofactor(Compound):
    __slots__ = ("_A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Cofactor of tensor with rank != 2 is undefined.")
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Cofactor.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return self._A.shape()
    
    #def as_basic(self, A):
    #    return FIXME
    
    def __str__(self):
        return "cofactor(%s)" % self._A
    
    def __repr__(self):
        return "Cofactor(%r)" % self._A

