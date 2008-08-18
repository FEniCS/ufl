"""Compound tensor algebra operations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-08-14"

from .output import ufl_assert
from .base import Terminal, UFLObject


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

class Transpose(UFLObject):
    __slots__ = ("_A",)
    
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Transpose is only defined for rank 2 tensors.")
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
        return "Transpose(%r)" % self._A

class Outer(UFLObject):
    __slots__ = ("a", "b", "_free_indices")

    def __init__(self, a, b):
        self.a = a
        self.b = b
        ai = self.a.free_indices()
        bi = self.b.free_indices()
        ufl_assert( len(set(ai) ^ set(bi)) == 0, "Didn't expect repeated indices in outer product.") 
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self.a, self.b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self.a.shape() + self.b.shape()
    
    def __str__(self):
        return "(%s) (x) (%s)" % (self.a, self.b)
        #return "%s (x) %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Outer(%r, %r)" % (self.a, self.b)

class Inner(UFLObject):
    __slots__ = ("a", "b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() == b.rank(), "Rank mismatch.")
        self.a = a
        self.b = b
        ai = self.a.free_indices()
        bi = self.b.free_indices()
        ufl_assert( len(set(ai) ^ set(bi)) == 0, "Didn't expect repeated indices in outer product.") 
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self.a, self.b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "(%s) : (%s)" % (self.a, self.b)
        #return "%s : %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Inner(%r, %r)" % (self.a, self.b)

class Dot(UFLObject):
    __slots__ = ("a", "b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() >= 1 and b.rank() >= 1, "Dot product requires arguments of rank >= 1, got %d and %d." % (a.rank(), b.rank())) # TODO: maybe scalars are ok?
        self.a = a
        self.b = b
        ai = self.a.free_indices()
        bi = self.b.free_indices()
        ufl_assert( len(set(ai) ^ set(bi)) == 0, "Didn't expect repeated indices in outer product.") 
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self.a, self.b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return self.a.shape()[:-1] + self.b.shape()[1:]
    
    def __str__(self):
        return "(%s) . (%s)" % (self.a, self.b)
        #return "%s . %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Dot(%r, %r)" % (self.a, self.b)

class Cross(UFLObject):
    __slots__ = ("a", "b", "_free_indices")

    def __init__(self, a, b):
        ufl_assert(a.rank() == 1 and b.rank() == 1, "Cross product requires arguments of rank 1.")
        self.a = a
        self.b = b
        ai = self.a.free_indices()
        bi = self.b.free_indices()
        ufl_assert( len(set(ai) ^ set(bi)) == 0, "Didn't expect repeated indices in outer product.") 
        self._free_indices = tuple(ai+bi)
    
    def operands(self):
        return (self.a, self.b)
    
    def free_indices(self):
        return self._free_indices
    
    def shape(self):
        return (3,)
    
    def __str__(self):
        return "(%s) x (%s)" % (self.a, self.b)
        #return "%s x %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Cross(%r, %r)" % (self.a, self.b)

class Trace(UFLObject):
    __slots__ = ("A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Trace of tensor with rank != 2 is undefined.")
        self.A = A
    
    def operands(self):
        return (self.A, )
    
    def free_indices(self):
        return self.A.free_indices()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "tr(%s)" % self.A
    
    def __repr__(self):
        return "Trace(%r)" % self.A

class Determinant(UFLObject):
    __slots__ = ("A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Determinant of tensor with rank != 2 is undefined.")
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in determinant.")
        self.A = A
    
    def operands(self):
        return (self.A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "det(%s)" % self.A
    
    def __repr__(self):
        return "Determinant(%r)" % self.A

class Inverse(UFLObject):
    __slots__ = ("A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Inverse of tensor with rank != 2 is undefined.")
        s = A.shape()
        ufl_assert(s[0] == s[1], "Cannot take inverse of rectangular matrix with dimensions %s." % repr(s))
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Inverse.")
        self.A = A
    
    def operands(self):
        return (self.A, )
    
    def free_indices(self):
        return ()
    
    def shape(self):
        return A.shape()
    
    def __str__(self):
        return "(%s)^-1" % self.A
    
    def __repr__(self):
        return "Inverse(%r)" % self.A

class Deviatoric(UFLObject):
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
    
    def __str__(self):
        return "dev(%s)" % self._A
    
    def __repr__(self):
        return "Deviatoric(%r)" % self._A

class Cofactor(UFLObject):
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
    
    def __str__(self):
        return "cofactor(%s)" % self._A
    
    def __repr__(self):
        return "Cofactor(%r)" % self._A

