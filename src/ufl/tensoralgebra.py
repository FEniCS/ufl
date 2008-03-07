#!/usr/bin/env python

"""
Compound tensor algebra operations. Needs some work!
"""

__authors__ = "Martin Sandve Alnes"
__date__ = "March 8th 2008"

from base import *

### Algebraic operations on tensors:
# TODO: define dot, inner, and contract clearly:
# Scalars:
#   dot(a,b)      = a*b
#   inner(a,b)    = a*b
#   contract(a,b) = a*b
# Vectors:
#   dot(u,v)      = u_i v_i
#   inner(u,v)    = u_i v_i
#   contract(u,v) = u_i v_i
# Matrices:
#   dot(A,B)      = A_{ik} B_{kj}
#   inner(A,B)    = A_{ij} B_{ij}
#   contract(A,B) = A_{ij} B_{ij}
# Combined:
#   dot(A,u)      = A_{ik} u_k
#   inner(A,u)    = A_{ik} u_k
#   contract(A,u) = A_{ik} u_k
#   dot(u,B)      = u_k B_{ki}
#   inner(u,B)    = u_k B_{ki}
#   contract(u,B) = u_k B_{ki}
#
# Maybe in general (contract is clearly a duplicate of inner above):
#   dot(x,y)   = contract(x, -1, y, 0)        # (last x dim) vs (first y dim)
#   inner(x,y) = contract(A, (0,1), B, (0,1)) # (all A dims) vs (all B dims)
#   contract(x,(xi),y,(yi)) = \sum_i x_{xi} y_{yi} # something like this, xi and yi are multiindices, TODO: need to design index stuff properly
#
#   dot(x,y): last index of x has same dimension as first index of y
#   inner(x,y): shape of x equals the shape of y
#   contract(x, xi, y, yi): len(xi) == len(yi), dimensions of indices in xi and yi match, dim(x) >= max(xi), dim(y) >= max(yi)

# objects representing the operations:

class Outer(UFLObject):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.free_indices = a.free_indices + b.free_indices
    
    def operands(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Outer(%s, %s)" % (repr(self.a), repr(self.b))

class Inner(UFLObject):
    def __init__(self, a, b):
        ufl_assert(a.rank() == b.rank(), "Rank mismatch.")
        self.a = a
        self.b = b
        self.free_indices = tuple()
    
    def operands(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Inner(%s, %s)" % (repr(self.a), repr(self.b))

class Dot(UFLObject):
    def __init__(self, a, b):
        #ufl_assert(a.rank() >= 1 and b.rank() >= 1, "Dot product requires arguments of rank >= 1.") # TODO: maybe scalars are ok?
        self.a = a
        self.b = b
        self.free_indices = a.free_indices[:-1] + b.free_indices[1:]
    
    def operands(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Dot(%s, %s)" % (self.a, self.b)

class Cross(UFLObject):
    def __init__(self, a, b):
        ufl_assert(a.rank() == 1 and b.rank() == 1, "Cross product requires arguments of rank 1.")
        self.a = a
        self.b = b
        self.free_indices = (Index(),)
    
    def operands(self):
        return (self.a, self.b)
    
    def __repr__(self):
        return "Cross(%s, %s)" % (repr(self.a), repr(self.b))

class Trace(UFLObject):
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Trace of tensor with rank != 2 is undefined.")
        self.A = A
        self.free_indices = tuple()
    
    def operands(self):
        return (self.A, )
    
    def __repr__(self):
        return "Trace(%s)" % repr(self.A)

class Determinant(UFLObject):
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Determinant of tensor with rank != 2 is undefined.")
        self.A = A
        self.free_indices = tuple()
    
    def operands(self):
        return (self.A, )
    
    def __repr__(self):
        return "Determinant(%s)" % repr(self.A)

class Inverse(UFLObject):
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Inverse of tensor with rank != 2 is undefined.")
        self.A = A
        self.free_indices = tuple(Index(), Index())
    
    def operands(self):
        return (self.A, )
    
    def __repr__(self):
        return "Inverse(%s)" % repr(self.A)

class Deviatoric(UFLObject):
    def __init__(self, A):
        ufl_assert(A.rank() == 2, "Deviatoric part of tensor with rank != 2 is undefined.")
        self.A = A
        self.free_indices = tuple(Index(), Index())
    
    def operands(self):
        return (self.A, )
    
    def __repr__(self):
        return "Deviatoric(%s)" % repr(self.A)

# functions exposed to the user:

def outer(a, b):
    return Outer(a, b)

def inner(a, b):
    return Inner(a, b)

def contract(a, b):
    return Contract(a, b)

def dot(a, b):
    return Dot(a, b)

def cross(a, b):
    return Cross(a, b)

def det(f):
    return Determinant(f)

def determinant(f):
    return Determinant(f)

def inverse(f):
    return Inverse(f)

def tr(f):
    return Trace(f)

def trace(f):
    return Trace(f)

def dev(A):
    return Deviatoric(A)

#def cofactor(A): # TODO:
#    return det(A)*inverse(A)

