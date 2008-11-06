"Basic algebra operations."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-20 -- 2008-11-06"

# Modified by Anders Logg, 2008

from itertools import chain

from .output import ufl_assert, ufl_error, ufl_warning
from .common import product, mergedicts
from .base import Expr
from .zero import Zero
from .scalar import ScalarValue, FloatValue, IntValue, is_true_ufl_scalar, is_python_scalar, as_ufl
from .indexing import extract_indices

#--- Algebraic operators ---

class Sum(Expr):
    __slots__ = ("_operands", "_repr")
    
    def __new__(cls, *operands): # TODO: This seems a bit complicated... Can it be simplified? Maybe we can merge some loops for efficiency?
        ufl_assert(len(operands), "Can't take sum of nothing.")
        
        # make sure everything is an Expr
        operands = [as_ufl(o) for o in operands]
        
        # assert consistent tensor properties
        sh = operands[0].shape()
        fi = operands[0].free_indices()
        ufl_assert(all(sh == o.shape() for o in operands[1:]),
            "Shape mismatch in Sum.")
        ufl_assert(all(fi == o.free_indices() for o in operands[1:]),
            "Can't add expressions with different free indices.")
        
        # purge zeros
        operands = [o for o in operands if not isinstance(o, Zero)]
        
        # sort operands by their repr TODO: This may be slow, can we do better? Needs to be completely independent of the outside world.
        operands = sorted(operands, key = lambda x: repr(x))
        
        # sort scalars to beginning and merge them
        scalars = [o for o in operands if isinstance(o, ScalarValue)]
        if scalars:
            # exploiting Pythons built-in coersion rules
            f = as_ufl(sum(f._value for f in scalars))
            nonscalars = [o for o in operands if not isinstance(o, ScalarValue)]
            if not nonscalars:
                return f
            if isinstance(f, Zero):
                operands = nonscalars
            else:
                operands = [f] + nonscalars
        
        # have we purged everything? 
        if not operands:
            ufl_warning("TODO: This shouldn't happen. Please email ufl-dev@fenics.org with an example that produces this message.")
            return Zero(sh)
        
        # left with one operand only?
        if len(operands) == 1:
            return operands[0]
        
        # Replace n-repeated operands foo with n*foo
        newoperands = []
        op = operands[0]
        n = 1
        for o in operands[1:] + [None]:
            if o == op:
                n += 1
            else:
                newoperands.append(op if n == 1 else n*op)
                op = o
                n = 1
        operands = newoperands
        
        # left with one operand only?
        if len(operands) == 1:
            return operands[0]
        
        # construct and initialize a new Sum object
        self = Expr.__new__(cls)
        self._init(*operands)
        return self

    def _init(self, *operands):
        ufl_assert(all(isinstance(o, Expr) for o in operands), "Expecting Expr instances.")
        self._operands = operands
        self._repr = "(%s)" % " + ".join(repr(o) for o in operands)
    
    def __init__(self, *operands):
        pass
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._operands[0].free_indices()
    
    def free_index_dimensions(self):
        return self._operands[0].free_index_dimensions()
    
    def shape(self):
        return self._operands[0].shape()
    
    def __str__(self):
        return "(%s)" % " + ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return self._repr

class Product(Expr):
    """The product of two or more UFL objects."""
    __slots__ = ("_operands", "_free_indices", "_free_index_dimensions", "_repeated_indices", "_shape", "_repr")
    
    def __new__(cls, *operands):
        ufl_assert(len(operands) >= 2, "Can't make product of nothing, should catch this before getting here.")

        # simplify if zero
        if any(o == 0 for o in operands):
            return Zero()

        operands = [as_ufl(o) for o in operands]
        
        # sort operands by their repr TODO: This may be slow, can we do better? Needs to be completely independent of the outside world.
        operands = sorted(operands, key = lambda x: repr(x))
        
        #ufl_assert(all(o.shape() == () for o in operands), "Expecting scalar valued operands.")
        # Get shape and move nonscalar operand to the end
        sh = ()
        j = None
        for i, o in enumerate(operands):
            sh2 = o.shape()
            if sh2 != ():
                ufl_assert(sh == (), "Found two nonscalar operands in Product, this is undefined.")
                sh = sh2
                j = i
        if j is not None:
            # We have a non-scalar expression in this product
            operands = operands[:j] + operands[j+1:] + [operands[j]]  
        
        # Replace n-repeated operands foo with foo**n
        newoperands = []
        op = operands[0]
        n = 1
        for o in operands[1:] + [None]:
            if o == op:
                n += 1
            else:
                newoperands.extend([op]*n if (n == 1 or op.free_indices()) else (op**n,))
                op = o
                n = 1
        operands = newoperands
        
        # left with one operand only?
        if len(operands) == 1:
            return operands[0]
        
        # merge scalars, but keep nonscalars sorted
        scalars = [o for o in operands if isinstance(o, ScalarValue)]
        if scalars:
            p = as_ufl(product(s._value for s in scalars))
            nonscalars = [o for o in operands if not isinstance(o, ScalarValue)]
            if not nonscalars:
                return p
            if p == 1:
                operands = nonscalars
            else:
                operands = [p] + nonscalars
        
        # left with one operand only?
        if len(operands) == 1:
            return operands[0]
        
        # construct and initialize a new Product object
        self = Expr.__new__(cls)
        self._init(sh, *operands)
        return self
    
    def _init(self, sh, *operands):
        ufl_assert(all(isinstance(o, Expr) for o in operands), "Expecting Expr instances.")
        self._operands = operands
        self._shape = sh
        
        # Extract indices
        all_indices = tuple(chain(*(o.free_indices() for o in operands)))
        all_index_dimensions = mergedicts([o.free_index_dimensions() for o in operands])
        (self._free_indices, self._repeated_indices, dummy, dummy) = \
            extract_indices(all_indices)
        self._free_index_dimensions = dict((i,all_index_dimensions[i]) for i in self._free_indices)
        
        self._repr = "(%s)" % " * ".join(repr(o) for o in self._operands)
    
    def __init__(self, *operands):
        pass
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._free_indices
    
    def free_index_dimensions(self):
        return self._free_index_dimensions
    
    def repeated_indices(self):
        return self._repeated_indices

    def shape(self):
        return self._shape
    
    def repeated_index_dimensions(self, default_dim):
        d = {}
        for i in self._repeated_indices:
            d[i] = default_dim # TODO: Allow other dimensions here!
        return d
    
    def __str__(self):
        return "(%s)" % " * ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return self._repr

class Division(Expr):
    __slots__ = ("_a", "_b")
    
    def __new__(cls, a, b):
        a = as_ufl(a)
        b = as_ufl(b)

        ufl_assert(b != 0, "Division by zero!")
        ufl_assert(is_true_ufl_scalar(b), "Division by non-scalar.")
        
        if isinstance(a, Zero):
            return a
        
        # TODO: Handling int/int specially here to avoid "2/3 == 0", do we want this?
        if isinstance(a, IntValue) and isinstance(b, IntValue):
            return as_ufl(a._value / float(b._value))
        
        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            return as_ufl(a._value / b._value)
        
        # construct and initialize a new Division object
        self = Expr.__new__(cls)
        self._init(a, b)
        return self
    
    def _init(self, a, b):
        ufl_assert(all(isinstance(o, Expr) for o in (a, b)), "Expecting Expr instances.")
        self._a = a
        self._b = b

    def __init__(self, a, b):
        pass
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._a.free_indices()
    
    def free_index_dimensions(self):
        return self._a.free_index_dimensions()
    
    def shape(self):
        return self._a.shape()

    def is_linear(self):
        return False
    
    def __str__(self):
        return "(%s / %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%r / %r)" % (self._a, self._b)

class Power(Expr):
    __slots__ = ("_a", "_b")
    
    def __new__(cls, a, b):
        a = as_ufl(a)
        b = as_ufl(b)
        if not (is_true_ufl_scalar(a) and is_true_ufl_scalar(b)):
            print 
            print "Non-scalar power error:"
            print a
            print b
            print 
        
        ufl_assert(is_true_ufl_scalar(a) and is_true_ufl_scalar(b),
            "Non-scalar power not defined.")
        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            return as_ufl(a._value ** b._value)
        if b == 1:
            return a
        if b == 0:
            return IntValue(1)
        
        # construct and initialize a new Power object
        self = Expr.__new__(cls)
        self._init(a, b)
        return self
    
    def _init(self, a, b):
        ufl_assert(all(isinstance(o, Expr) for o in (a, b)), "Expecting Expr instances.")
        self._a = a
        self._b = b

    def __init__(self, a, b):
        pass
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return ()
    
    def free_index_dimensions(self):
        return {}
    
    def shape(self):
        return ()

    def is_linear(self):
        return isinstance(self._b, int)
    
    def __str__(self):
        return "(%s ** %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return "(%r ** %r)" % (self._a, self._b)

class Abs(Expr):
    __slots__ = ("_a",)
    
    def __init__(self, a):
        ufl_assert(isinstance(a, Expr), "Expecting Expr instance.")
        self._a = a
    
    def operands(self):
        return (self._a, )
    
    def free_indices(self):
        return self._a.free_indices()
    
    def free_index_dimensions(self):
        return self._a.free_index_dimensions()
    
    def shape(self):
        return self._a.shape()

    def is_linear(self):
        return False
    
    def __str__(self):
        return "| %s |" % str(self._a)
    
    def __repr__(self):
        return "Abs(%r)" % self._a
