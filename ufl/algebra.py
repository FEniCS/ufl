"""Basic algebra operations."""

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-20 -- 2008-05-20"

from output import *
from base import *
from indexing import *


#--- Algebraic operators ---

class Sum(UFLObject):
    __slots__ = ("_operands",)
    
    def __init__(self, *operands):
        ufl_assert(len(operands), "Got sum of nothing.")
        ufl_assert(all(operands[0].rank()         == o.rank()         for o in operands), "Rank mismatch in sum.")
        ufl_assert(all(operands[0].free_indices() == o.free_indices() for o in operands), "Can't add expressions with different free indices.")
        self._operands = tuple(operands)
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._operands[0].free_indices()
    
    def rank(self):
        return self._operands[0].rank()
    
    def __str__(self):
        return "(%s)" % " + ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " + ".join(repr(o) for o in self._operands)

class Product(UFLObject):
    __slots__ = ("_operands", "_rank", "_free_indices", "_repeated_indices")
    
    def __init__(self, *operands):
        ufl_assert(len(operands), "Got product of nothing.")
        self._operands = tuple(operands)
       
        # Extract indices
        all_indices = tuple(chain(*(o.free_indices() for o in operands)))
        (fixed_indices, free_indices, repeated_indices, num_unassigned_indices) = extract_indices(all_indices)
        self._free_indices       = free_indices
        self._repeated_indices   = repeated_indices

        # Try to determine rank of this sequence of
        # products with possibly varying ranks of each operand.
        # Products currently defined as valid are:
        # - something multiplied with a scalar
        # - a scalar multiplied with something
        # - matrix-matrix (A*B, M*grad(u))
        # - matrix-vector (A*v)
        current_rank = operands[0].rank()
        for o in operands[1:]:
            if current_rank == 0 or o.rank() == 0:
                # at least one scalar
                current_rank = current_rank + o.rank()
            elif current_rank == 2 and o.rank() == 2:
                # matrix-matrix product
                current_rank = 2
            elif current_rank == 2 and o.rank() == 1:
                # matrix-vector product
                current_rank = 1
            else:
                ufl_error("Invalid combination of tensor ranks in product.")
        
        self._rank = current_rank
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._free_indices
    
    def rank(self):
        return self._rank
    
    def __str__(self):
        return "(%s)" % " * ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return "(%s)" % " * ".join(repr(o) for o in self._operands)

class Division(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(b), "Division by non-scalar.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._a.free_indices()
    
    def rank(self):
        return self._a.rank()
    
    def __str__(self):
        return "(%s / %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%s / %s)" % (repr(self._a), repr(self._b))

class Power(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar power not defined.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "(%s ** %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%s ** %s)" % (repr(self._a), repr(self._b))

class Mod(UFLObject):
    __slots__ = ("_a", "_b")
    
    def __init__(self, a, b):
        ufl_assert(is_true_scalar(a) and is_true_scalar(b), "Non-scalar mod not defined.")
        self._a = a
        self._b = b
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return tuple()
    
    def rank(self):
        return 0
    
    def __str__(self):
        return "(%s %% %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%s %% %s)" % (repr(self._a), repr(self._b))

class Abs(UFLObject):
    __slots__ = ("_a",)
    
    def __init__(self, a):
        self._a = a
    
    def operands(self):
        return (self._a, )
    
    def free_indices(self):
        return self._a.free_indices()
    
    def rank(self):
        return self._a.rank()
    
    def __str__(self):
        return "| %s |" % str(self._a)
    
    def __repr__(self):
        return "Abs(%s)" % repr(self._a)



# Extend UFLObject with algebraic operators

def _add(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Sum(self, o)
UFLObject.__add__ = _add

def _radd(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Sum(o, self)
UFLObject.__radd__ = _radd

def _sub(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return self + (-o)
UFLObject.__sub__ = _sub

def _rsub(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return o + (-self)
UFLObject.__rsub__ = _rsub

def _mul(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Product(self, o)
UFLObject.__mul__ = _mul

def _rmul(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Product(o, self)
UFLObject.__rmul__ = _rmul

def _div(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Division(self, o)
UFLObject.__div__ = _div

def _rdiv(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Division(o, self)
UFLObject.__rdiv__ = _rdiv

def _pow(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Power(self, o)
UFLObject.__pow__ = _pow

def _rpow(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Power(o, self)
UFLObject.__rpow__ = _rpow

def _mod(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Mod(self, o)
UFLObject.__mod__ = _mod

def _rmod(self, o):
    if is_python_scalar(o): o = Number(o)
    if not isinstance(o, UFLObject): return NotImplemented
    return Mod(o, self)
UFLObject.__rmod__ = _rmod

def _neg(self):
    return -1*self
UFLObject.__neg__ = _neg

def _abs(self):
    return Abs(self)
UFLObject.__abs__ = _abs

