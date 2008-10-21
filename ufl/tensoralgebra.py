"""Compound tensor algebra operations."""

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-10-21"

from .output import ufl_assert
from .base import UFLObject, Terminal, Compound
from .indexing import Index, indices, compare_shapes
from .tensors import as_tensor


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

#
# Note:
# To avoid typing errors, the expressions for cofactor and deviatoric parts in as_basic
# below were created with the script tensoralgebrastrings.py under ufl/scripts/
#

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
    
    def as_basic(self, dim, A):
        ii = Index()
        jj = Index()
        return as_tensor(A[ii, jj], (jj, ii))
    
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

    def as_basic(self, dim, a, b):
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
    
    def as_basic(self, dim, a, b):
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
        ufl_assert(a.rank() >= 1 and b.rank() >= 1,
            "Dot product requires arguments of rank >= 1, got %d and %d." % \
            (a.rank(), b.rank())) # TODO: maybe scalars are ok?
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
        return self._a.shape()[:-1] + self._b.shape()[1:]
    
    def as_basic(self, dim, a, b):
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
    
    def as_basic(self, dim, a, b):
        if dim == 3:
            ufl_assert(compare_shapes(a.shape(), (3,)),
                "Invalid shape of first argument in cross product.")
            ufl_assert(compare_shapes(b.shape(), (3,)),
                "Invalid shape of second argument in cross product.")
            def c(i, j):
                return a[i]*b[j]-a[j]*b[i]
            return as_vector(c(1,2), c(2,0), c(0,1))
        ufl_error("Cross product not implemented for dimension %d." % dim)
    
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
    
    def as_basic(self, dim, A):
        i = Index()
        return A[i,i]
    
    def __str__(self):
        return "tr(%s)" % self._A
    
    def __repr__(self):
        return "Trace(%r)" % self._A


class Determinant(Compound):
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
    
    def as_basic(self, dim, A):
        sh = complete_shape(A.shape(), dim)

        if len(sh) == 0:
            return A
        
        def det2D(B, i, j, k, l):
            return B[i,k]*B[j,l]-B[i,l]*B[j,k]

        if sh[0] == 2:
            return det2D(A, 0, 1, 0, 1)
        
        if sh[0] == 3:
            # TODO: Verify this expression
            return A[0,0]*det2D(A, 1, 2, 1, 2) + \
                   A[0,1]*det2D(A, 1, 2, 2, 0) + \
                   A[0,2]*det2D(A, 1, 2, 0, 1)
        
        # TODO: Implement generally for all dimensions?
        ufl_error("Determinant not implemented for dimension %d." % dim)
    
    def __str__(self):
        return "det(%s)" % self._A
    
    def __repr__(self):
        return "Determinant(%r)" % self._A


class Inverse(Compound): # TODO: Drop Inverse and represent it as product of Determinant and Cofactor?
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
    
    def as_basic(self, dim, A):
        if A.rank() == 0:
            return 1.0 / A
        return Determinant(A).as_basic(dim, A) * Cofactor(A).as_basic(dim, A)
    
    def __str__(self):
        return "(%s)^-1" % self._A
    
    def __repr__(self):
        return "Inverse(%r)" % self._A


class Cofactor(Compound):
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

    def as_basic(self, dim, A):
        sh = complete_shape(A.shape(), dim)
        if sh[0] == 2:
            return as_matrix([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([[A[2,2]*A[1,1]-A[1,2]*A[2,1],-A[0,1]*A[2,2]+A[0,2]*A[2,1],A[0,1]*A[1,2]-A[0,2]*A[1,1]],[-A[2,2]*A[1,0]+A[1,2]*A[2,0],-A[0,2]*A[2,0]+A[2,2]*A[0,0],A[0,2]*A[1,0]-A[1,2]*A[0,0]],[A[1,0]*A[2,1]-A[2,0]*A[1,1],A[0,1]*A[2,0]-A[0,0]*A[2,1],A[0,0]*A[1,1]-A[0,1]*A[1,0]]])
        elif sh[0] == 4:
            return as_matrix([[-A[3,3]*A[2,1]*A[1,2]+A[1,2]*A[3,1]*A[2,3]+A[1,1]*A[3,3]*A[2,2]-A[3,1]*A[2,2]*A[1,3]+A[2,1]*A[1,3]*A[3,2]-A[1,1]*A[3,2]*A[2,3],-A[3,1]*A[0,2]*A[2,3]+A[0,1]*A[3,2]*A[2,3]-A[0,3]*A[2,1]*A[3,2]+A[3,3]*A[2,1]*A[0,2]-A[3,3]*A[0,1]*A[2,2]+A[0,3]*A[3,1]*A[2,2],A[3,1]*A[1,3]*A[0,2]+A[1,1]*A[0,3]*A[3,2]-A[0,3]*A[1,2]*A[3,1]-A[0,1]*A[1,3]*A[3,2]+A[3,3]*A[1,2]*A[0,1]-A[1,1]*A[3,3]*A[0,2],A[1,1]*A[0,2]*A[2,3]-A[2,1]*A[1,3]*A[0,2]+A[0,3]*A[2,1]*A[1,2]-A[1,2]*A[0,1]*A[2,3]-A[1,1]*A[0,3]*A[2,2]+A[0,1]*A[2,2]*A[1,3]],[A[3,3]*A[1,2]*A[2,0]-A[3,0]*A[1,2]*A[2,3]+A[1,0]*A[3,2]*A[2,3]-A[3,3]*A[1,0]*A[2,2]-A[1,3]*A[3,2]*A[2,0]+A[3,0]*A[2,2]*A[1,3],A[0,3]*A[3,2]*A[2,0]-A[0,3]*A[3,0]*A[2,2]+A[3,3]*A[0,0]*A[2,2]+A[3,0]*A[0,2]*A[2,3]-A[0,0]*A[3,2]*A[2,3]-A[3,3]*A[0,2]*A[2,0],-A[3,3]*A[0,0]*A[1,2]+A[0,0]*A[1,3]*A[3,2]-A[3,0]*A[1,3]*A[0,2]+A[3,3]*A[1,0]*A[0,2]+A[0,3]*A[3,0]*A[1,2]-A[0,3]*A[1,0]*A[3,2],A[0,3]*A[1,0]*A[2,2]+A[1,3]*A[0,2]*A[2,0]-A[0,0]*A[2,2]*A[1,3]-A[0,3]*A[1,2]*A[2,0]+A[0,0]*A[1,2]*A[2,3]-A[1,0]*A[0,2]*A[2,3]],[A[3,1]*A[1,3]*A[2,0]+A[3,3]*A[2,1]*A[1,0]+A[1,1]*A[3,0]*A[2,3]-A[1,0]*A[3,1]*A[2,3]-A[3,0]*A[2,1]*A[1,3]-A[1,1]*A[3,3]*A[2,0],A[3,3]*A[0,1]*A[2,0]-A[3,3]*A[0,0]*A[2,1]-A[0,3]*A[3,1]*A[2,0]-A[3,0]*A[0,1]*A[2,3]+A[0,0]*A[3,1]*A[2,3]+A[0,3]*A[3,0]*A[2,1],-A[0,0]*A[3,1]*A[1,3]+A[0,3]*A[1,0]*A[3,1]-A[3,3]*A[1,0]*A[0,1]+A[1,1]*A[3,3]*A[0,0]-A[1,1]*A[0,3]*A[3,0]+A[3,0]*A[0,1]*A[1,3],A[0,0]*A[2,1]*A[1,3]+A[1,0]*A[0,1]*A[2,3]-A[0,3]*A[2,1]*A[1,0]+A[1,1]*A[0,3]*A[2,0]-A[1,1]*A[0,0]*A[2,3]-A[0,1]*A[1,3]*A[2,0]],[-A[1,2]*A[3,1]*A[2,0]-A[2,1]*A[1,0]*A[3,2]+A[3,0]*A[2,1]*A[1,2]-A[1,1]*A[3,0]*A[2,2]+A[1,0]*A[3,1]*A[2,2]+A[1,1]*A[3,2]*A[2,0],-A[3,0]*A[2,1]*A[0,2]-A[0,1]*A[3,2]*A[2,0]+A[3,1]*A[0,2]*A[2,0]-A[0,0]*A[3,1]*A[2,2]+A[3,0]*A[0,1]*A[2,2]+A[0,0]*A[2,1]*A[3,2],A[0,0]*A[1,2]*A[3,1]-A[1,0]*A[3,1]*A[0,2]+A[1,1]*A[3,0]*A[0,2]+A[1,0]*A[0,1]*A[3,2]-A[3,0]*A[1,2]*A[0,1]-A[1,1]*A[0,0]*A[3,2],-A[1,1]*A[0,2]*A[2,0]+A[2,1]*A[1,0]*A[0,2]+A[1,2]*A[0,1]*A[2,0]+A[1,1]*A[0,0]*A[2,2]-A[1,0]*A[0,1]*A[2,2]-A[0,0]*A[2,1]*A[1,2]]])
        ufl_error("Cofactor not implemented for dimension %s." % sh[0])

    def __str__(self):
        return "cofactor(%s)" % self._A
    
    def __repr__(self):
        return "Cofactor(%r)" % self._A


class Deviatoric(Compound):
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
    
    def as_basic(self, dim, A):
        sh = complete_shape(A.shape(), dim)
        if sh[0] == 2:
            return as_matrix([[-A[1,1],A[0,1]],[A[1,0],-A[0,0]]])
        elif sh[0] == 3:
            return as_matrix([[-A[1,1]-A[2,2],A[0,1],A[0,2]],[A[1,0],-A[0,0]-A[2,2],A[1,2]],[A[2,0],A[2,1],-A[0,0]-A[1,1]]])
        ufl_error("dev(A) not implemented for dimension %s." % sh[0])
    
    def __str__(self):
        return "dev(%s)" % self._A
    
    def __repr__(self):
        return "Deviatoric(%r)" % self._A


class Skew(Compound):
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
    
    def as_basic(self, dim, A):
        i, j = indices(2)
        return as_matrix( (A[i,j] - A[j,i]) / 2, (i,j) )
    
    def __str__(self):
        return "skew(%s)" % self._A
    
    def __repr__(self):
        return "Skew(%r)" % self._A

