#!/usr/bin/env python

"""
Compound tensor algebra operations. Needs some work!
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-04-01"

from output import *
from base import *

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



# objects representing the operations:

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
    
    def rank(self):
        return self.a.rank() + self.b.rank()
    
    def __str__(self):
        return "(%s) (x) (%s)" % (str(self.a), str(self.b))
        #return "%s (x) %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Outer(%s, %s)" % (repr(self.a), repr(self.b))

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
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "(%s) : (%s)" % (pstr(self.a, self), pstr(self.b, self))
        #return "%s : %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Inner(%s, %s)" % (repr(self.a), repr(self.b))

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
    
    def rank(self):
        return self.a.rank() + self.b.rank() - 2
    
    def __str__(self):
        return "(%s) . (%s)" % (str(self.a), str(self.b))
        #return "%s . %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Dot(%s, %s)" % (self.a, self.b)

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
    
    def rank(self):
        return 1
    
    def __str__(self):
        return "(%s) x (%s)" % (str(self.a), str(self.b))
        #return "%s x %s" % (pstr(self.a, self), pstr(self.b, self))
    
    def __repr__(self):
        return "Cross(%s, %s)" % (repr(self.a), repr(self.b))

class Trace(UFLObject):
    __slots__ = ("A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Trace of tensor with rank != 2 is undefined.")
        self.A = A
    
    def operands(self):
        return (self.A, )
    
    def free_indices(self):
        return self.A.free_indices()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "tr(%s)" % str(self.A)
    
    def __repr__(self):
        return "Trace(%s)" % repr(self.A)

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
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "det(%s)" % str(self.A)
    
    def __repr__(self):
        return "Determinant(%s)" % repr(self.A)

class Inverse(UFLObject):
    __slots__ = ("A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Inverse of tensor with rank != 2 is undefined.")
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Inverse.")
        self.A = A
    
    def operands(self):
        return (self.A, )
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return 2
    
    def __str__(self):
        return "(%s)^-1" % str(self.A)
    
    def __repr__(self):
        return "Inverse(%s)" % repr(self.A)

class Deviatoric(UFLObject):
    __slots__ = ("A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Deviatoric part of tensor with rank != 2 is undefined.")
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Deviatoric.")
        self.A = A
    
    def operands(self):
        return (self.A, )
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return 2
    
    def __str__(self):
        return "dev(%s)" % str(self.A)
    
    def __repr__(self):
        return "Deviatoric(%s)" % repr(self.A)

class Cofactor(UFLObject):
    __slots__ = ("A",)

    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Cofactor of tensor with rank != 2 is undefined.")
        ufl_assert(len(A.free_indices()) == 0, "Didn't expect free indices in Cofactor.")
        self.A = A
    
    def operands(self):
        return (self.A, )
    
    def free_indices(self):
        return ()
    
    def rank(self):
        return 2
    
    def __str__(self):
        return "cofactor(%s)" % str(self.A)
    
    def __repr__(self):
        return "Cofactor(%s)" % repr(self.A)


# functions exposed to the user:

def outer(a, b):
    return Outer(a, b)

def inner(a, b):
    return Inner(a, b)

def dot(a, b):
    return Dot(a, b)

def cross(a, b):
    return Cross(a, b)

def det(f):
    return Determinant(f)

def inv(f):
    return Inverse(f)

def tr(f):
    return Trace(f)

def dev(A):
    return Deviatoric(A)

def cofac(A):
    return Cofactor(A)
