"""Compound tensor algebra operations."""


__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-14 -- 2008-11-06"

from ufl.output import ufl_assert, ufl_warning
from ufl.base import Expr, Terminal
from ufl.zero import Zero
from ufl.indexing import Index, indices

def merge_free_indices(a, b):
    ai = a.free_indices()
    bi = b.free_indices()
    ufl_assert(not (set(ai) ^ set(bi)), "Not expecting repeated indices.")
    free_indices = ai+bi
    
    aid = a.index_dimensions()
    bid = b.index_dimensions()
    index_dimensions = dict((i, aid[i]) for i in ai)
    index_dimensions.update((i, bid[i]) for i in bi)
    
    return free_indices, index_dimensions 

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
        Terminal.__init__(self)
        self._dim = dim
    
    def shape(self):
        return (self._dim, self._dim)
    
    def __str__(self):
        return "I"
    
    def __repr__(self):
        return "Identity(%d)" % self._dim
    
    def __eq__(self, other):
        return isinstance(other, Identity) and self._dim == other._dim


# objects representing the operations:

class Transposed(Expr):
    __slots__ = ("_A",)

    def __new__(cls, A):
        if isinstance(A, Zero):
            a, b = A.shape()
            return Zero((b, a), A.free_indices(), A.index_dimensions())
        return Terminal.__new__(cls)
    
    def __init__(self, A):
        Expr.__init__(self)
        ufl_assert(A.rank() == 2, "Transposed is only defined for rank 2 tensors.")
        self._A = A
    
    def operands(self):
        return (self._A,)
    
    def free_indices(self):
        return self._A.free_indices()
    
    def index_dimensions(self):
        return self._A.index_dimensions()
    
    def shape(self):
        s = self._A.shape()
        return (s[1], s[0])
    
    def __str__(self):
        return "(%s)^T" % self._A
    
    def __repr__(self):
        return "Transposed(%r)" % self._A

class Outer(Expr):
    __slots__ = ("_a", "_b", "_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        if isinstance(a, Zero) or isinstance(b, Zero):
            free_indices, index_dimensions = merge_free_indices(a, b)
            return Zero(a.shape() + b.shape(), free_indices, index_dimensions)
        return Terminal.__new__(cls)

    def __init__(self, a, b):
        Expr.__init__(self)
        self._a = a
        self._b = b
        self._free_indices, self._index_dimensions = merge_free_indices(a, b)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._a.shape() + self._b.shape()
    
    def __str__(self):
        return "(%s) (x) (%s)" % (self._a, self._b)
        #return "%s (x) %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Outer(%r, %r)" % (self._a, self._b)

class Inner(Expr):
    __slots__ = ("_a", "_b", "_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        ufl_assert(a.shape() == b.shape(), "Shape mismatch.")
        if isinstance(a, Zero) or isinstance(b, Zero):
            free_indices, index_dimensions = merge_free_indices(a, b)
            return Zero((), free_indices, index_dimensions)
        return Terminal.__new__(cls)

    def __init__(self, a, b):
        Expr.__init__(self)
        # sort operands by their repr TODO: This may be slow, can we do better? Needs to be completely independent of the outside world.
        a, b = sorted((a,b), key = lambda x: repr(x))
        self._a = a
        self._b = b
        self._free_indices, self._index_dimensions = merge_free_indices(a, b)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "(%s) : (%s)" % (self._a, self._b)
        #return "%s : %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Inner(%r, %r)" % (self._a, self._b)

class Dot(Expr):
    __slots__ = ("_a", "_b", "_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        ufl_assert(a.rank() >= 1 and b.rank() >= 1,
            "Dot product requires arguments of rank >= 1, got %d and %d." % \
            (a.rank(), b.rank())) # TODO: maybe scalars are ok?
        
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = a.shape()[:-1] + b.shape()[1:]
            free_indices, index_dimensions = merge_free_indices(a, b)
            return Zero(shape, free_indices, index_dimensions)
        
        return Terminal.__new__(cls)

    def __init__(self, a, b):
        Expr.__init__(self)
        self._a = a
        self._b = b
        self._free_indices, self._index_dimensions = merge_free_indices(a, b)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._a.shape()[:-1] + self._b.shape()[1:]
    
    def __str__(self):
        return "(%s) . (%s)" % (self._a, self._b)
        #return "%s . %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Dot(%r, %r)" % (self._a, self._b)

class Cross(Expr):
    __slots__ = ("_a", "_b", "_free_indices", "_index_dimensions")

    def __new__(cls, a, b):
        if isinstance(a, Zero) or isinstance(b, Zero):
            ufl_warning("Returning zero from Cross not implemented.")
        return Terminal.__new__(cls)

    def __init__(self, a, b):
        Expr.__init__(self)
        ufl_assert(a.rank() == 1 and b.rank() == 1,
            "Cross product requires arguments of rank 1.")
        self._a = a
        self._b = b
        ai = self._a.free_indices()
        bi = self._b.free_indices()
        ufl_assert(not (set(ai) ^ set(bi)),
            "Not expecting repeated indices in outer product.") 
        self._free_indices = ai+bi
        self._index_dimensions = dict(a.index_dimensions())
        self._index_dimensions.update(b.index_dimensions())
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return (3,)
    
    def __str__(self):
        return "(%s) x (%s)" % (self._a, self._b)
        #return "%s x %s" % (pstr(self._a, self), pstr(self._b, self))
    
    def __repr__(self):
        return "Cross(%r, %r)" % (self._a, self._b)

class Trace(Expr):
    __slots__ = ("_A",)

    def __new__(cls, A):
        ufl_assert(A.rank() == 2, "Trace of tensor with rank != 2 is undefined.")
        if isinstance(A, Zero):
            return Zero((), A.free_indices(), A.index_dimensions())
        return Terminal.__new__(cls)

    def __init__(self, A):
        Expr.__init__(self)
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return self._A.free_indices()
    
    def index_dimensions(self):
        return self._A.index_dimensions()
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "tr(%s)" % self._A
    
    def __repr__(self):
        return "Trace(%r)" % self._A

class Determinant(Expr):
    __slots__ = ("_A",)

    def __new__(cls, A):
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 0 or r == 2,
            "Determinant of tensor with rank != 2 is undefined.")
        ufl_assert(r == 0 or sh[0] == sh[1],
            "Cannot take determinant of rectangular rank 2 tensor.")
        ufl_assert(not A.free_indices(),
            "Not expecting free indices in determinant.")
        if isinstance(A, Zero):
            return Zero((), A.free_indices(), A.index_dimensions())
        return Terminal.__new__(cls)

    def __init__(self, A):
        Expr.__init__(self)
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def index_dimensions(self):
        return {}
    
    def shape(self):
        return ()
    
    def __str__(self):
        return "det(%s)" % self._A
    
    def __repr__(self):
        return "Determinant(%r)" % self._A

class Inverse(Expr): # TODO: Drop Inverse and represent it as product of Determinant and Cofactor?
    __slots__ = ("_A",)

    def __init__(self, A):
        Expr.__init__(self)
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 0 or r == 2, "Inverse of tensor with rank != 2 or 0 is undefined.")
        ufl_assert(r == 0 or sh[0] == sh[1],
            "Cannot take inverse of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Inverse.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def index_dimensions(self):
        return {}
    
    def shape(self):
        return self._A.shape()
    
    def __str__(self):
        return "(%s)^-1" % self._A
    
    def __repr__(self):
        return "Inverse(%r)" % self._A

class Cofactor(Expr):
    __slots__ = ("_A",)

    def __init__(self, A):
        Expr.__init__(self)
        sh = A.shape()
        ufl_assert(len(sh) == 2, "Cofactor of tensor with rank != 2 is undefined.")
        ufl_assert(sh[0] == sh[1], "Cannot take cofactor of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Cofactor.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return ()
    
    def index_dimensions(self):
        return {}
    
    def shape(self):
        return self._A.shape()

    def __str__(self):
        return "cofactor(%s)" % self._A
    
    def __repr__(self):
        return "Cofactor(%r)" % self._A

class Deviatoric(Expr):
    __slots__ = ("_A",)

    def __init__(self, A):
        Expr.__init__(self)
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 2, "Deviatoric part of tensor with rank != 2 is undefined.")
        ufl_assert(sh[0] == sh[1],
            "Cannot take deviatoric part of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Deviatoric.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return self._A.free_indices()
    
    def index_dimensions(self):
        return self._A.index_dimensions()
    
    def shape(self):
        return self._A.shape()
    
    def __str__(self):
        return "dev(%s)" % self._A
    
    def __repr__(self):
        return "Deviatoric(%r)" % self._A

class Skew(Expr):
    __slots__ = ("_A",)

    def __init__(self, A):
        Expr.__init__(self)
        sh = A.shape()
        r = len(sh)
        ufl_assert(r == 2, "Skew part of tensor with rank != 2 is undefined.")
        ufl_assert(sh[0] == sh[1],
            "Cannot take skew part of rectangular matrix with dimensions %s." % repr(sh))
        ufl_assert(not A.free_indices(), "Not expecting free indices in Skew.")
        self._A = A
    
    def operands(self):
        return (self._A, )
    
    def free_indices(self):
        return self._A.free_indices()
    
    def index_dimensions(self):
        return self._A.index_dimensions()
    
    def shape(self):
        return self._A.shape()
    
    def __str__(self):
        return "skew(%s)" % self._A
    
    def __repr__(self):
        return "Skew(%r)" % self._A
