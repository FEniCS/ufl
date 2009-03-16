"Basic algebra operations."

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-05-20 -- 2009-02-20"

# Modified by Anders Logg, 2008

from collections import defaultdict
from itertools import chain

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import product, mergedicts, subdict
from ufl.expr import Expr, AlgebraOperator
from ufl.constantvalue import Zero, ScalarValue, FloatValue, IntValue, is_ufl_scalar, is_true_ufl_scalar, is_python_scalar, as_ufl
from ufl.indexing import IndexBase, Index, FixedIndex
from ufl.indexutils import unique_indices
from ufl.sorting import cmp_expr

#--- Algebraic operators ---

class Sum(AlgebraOperator):
    __slots__ = ("_operands", "_repr")
    
    def __new__(cls, *operands): # TODO: This seems a bit complicated... Can it be simplified? Maybe we can merge some loops for efficiency?
        #len(operands)  or  error("Can't take sum of nothing.")
        if not operands:
            return Zero()
        
        # make sure everything is an Expr
        operands = [as_ufl(o) for o in operands]
        
        # Got one operand only? Do nothing then.
        if len(operands) == 1:
            return operands[0]
        
        # assert consistent tensor properties
        sh = operands[0].shape()
        fi = operands[0].free_indices()
        fid = operands[0].index_dimensions()
        #ufl_assert(all(sh == o.shape() for o in operands[1:]),
        #    "Shape mismatch in Sum.")
        #ufl_assert(not any((set(fi) ^ set(o.free_indices())) for o in operands[1:]),
        #    "Can't add expressions with different free indices.")
        all(sh == o.shape() for o in operands[1:])\
            or error("Shape mismatch in Sum.")
        (not any((set(fi) ^ set(o.free_indices())) for o in operands[1:]))\
            or error("Can't add expressions with different free indices.")
        
        # sort operands in a canonical order
        operands = sorted(operands, cmp=cmp_expr)
        
        # purge zeros
        operands = [o for o in operands if not isinstance(o, Zero)]
        
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
            return Zero(sh, fi, fid)
        
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
        self = AlgebraOperator.__new__(cls)
        self._init(*operands)
        return self

    def _init(self, *operands):
        self._operands = operands
        self._repr = "Sum(%s)" % ", ".join(repr(o) for o in operands)
    
    def __init__(self, *operands):
        AlgebraOperator.__init__(self)
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._operands[0].free_indices()
    
    def index_dimensions(self):
        return self._operands[0].index_dimensions()
    
    def shape(self):
        return self._operands[0].shape()
    
    def evaluate(self, x, mapping, component, index_values):
        return sum(o.evaluate(x, mapping, component, index_values) for o in self.operands())
    
    def __str__(self):
        return "(%s)" % " + ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return self._repr

class Product(AlgebraOperator):
    """The product of two or more UFL objects."""
    __slots__ = ("_operands", "_free_indices", "_index_dimensions", "_repr")
    
    def __new__(cls, *operands):
        # Make sure everything is an Expr
        operands = [as_ufl(o) for o in operands]
        
        # Make sure everything is scalar
        #ufl_assert(not any(o.shape() for o in operands),
        #    "Product can only represent products of scalars.")
        not any(o.shape() for o in operands)\
            or error("Product can only represent products of scalars.")
        
        # No operands? Return one.
        if not operands:
            return IntValue(1)
        
        # Got one operand only? Just return it.
        if len(operands) == 1:
            return operands[0]
        
        # Got any zeros? Return zero.
        if any(isinstance(o, Zero) for o in operands):
            free_indices     = unique_indices(tuple(chain(*(o.free_indices() for o in operands))))
            index_dimensions = subdict(mergedicts([o.index_dimensions() for o in operands]), free_indices)
            return Zero((), free_indices, index_dimensions)
        
        # Merge scalars, but keep nonscalars sorted
        scalars = []
        nonscalars = []
        for o in operands:
            if isinstance(o, ScalarValue):
                scalars.append(o)
            else:
                nonscalars.append(o)
        if scalars:
            # merge scalars
            p = as_ufl(product(s._value for s in scalars))
            # only scalars?
            if not nonscalars:
                return p
            # merged scalar is unity?
            if p == 1:
                scalars = []
                # Left with one nonscalar operand only after merging scalars?
                if len(nonscalars) == 1:
                    return nonscalars[0]
            else:
                scalars = [p]
        
        # Sort operands in a canonical order (NB! This is fragile! Small changes here can have large effects.)
        operands = scalars + sorted(nonscalars, cmp=cmp_expr)
        
        # Replace n-repeated operands foo with foo**n
        newoperands = []
        op, nop = operands[0], 1
        for o in operands[1:] + [None]:
            if o == op:
                # op is repeated, count number of repetitions
                nop += 1
            else:
                if nop == 1:
                    # op is not repeated
                    newoperands.append(op)
                elif op.free_indices():
                    # We can't simplify products to powers if the operands has
                    # free indices, because of complications in differentiation.
                    # op repeated, but has free indices, so we don't simplify
                    newoperands.extend([op]*nop)
                else:
                    # op repeated, make it a power
                    newoperands.append(op**nop)
                # Reset op as o
                op, nop = o, 1
        operands = newoperands
        
        # Left with one operand only after simplifications?
        if len(operands) == 1:
            return operands[0]
        
        # Construct and initialize a new Product object
        self = AlgebraOperator.__new__(cls)
        self._init(*operands)
        return self
    
    def _init(self, *operands):
        "Constructor, called by __new__ with already checked arguments."
        # Store basic properties
        self._operands = operands
        
        # Extract indices
        self._free_indices     = unique_indices(tuple(chain(*(o.free_indices() for o in operands))))
        self._index_dimensions = mergedicts([o.index_dimensions() for o in operands])
        
        self._repr = "Product(%s)" % ", ".join(repr(o) for o in self._operands)
    
    def __init__(self, *operands):
        AlgebraOperator.__init__(self)
    
    def operands(self):
        return self._operands
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return ()
    
    def evaluate(self, x, mapping, component, index_values):
        ops = self.operands()
        sh = self.shape()
        if sh:
            ufl_assert(sh == ops[-1].shape(), "Expecting nonscalar product operand to be the last by convention.")
            tmp = ops[-1].evaluate(x, mapping, component, index_values)
            ops = ops[:-1]
        else:
            tmp = 1
        for o in ops:
            tmp *= o.evaluate(x, mapping, (), index_values)
        return tmp
    
    def __str__(self):
        return "(%s)" % " * ".join(str(o) for o in self._operands)
    
    def __repr__(self):
        return self._repr

class Division(AlgebraOperator):
    __slots__ = ("_a", "_b", "_repr")
    
    def __new__(cls, a, b):
        a = as_ufl(a)
        b = as_ufl(b)

        #ufl_assert(b != 0, "Division by zero!")
        #ufl_assert(is_true_ufl_scalar(b), "Division by non-scalar.")
        (b != 0) or error("Division by zero!")
        is_true_ufl_scalar(b) or error("Division by non-scalar.")
        
        if isinstance(a, Zero):
            return a
        
        # TODO: Handling int/int specially here to avoid "2/3 == 0", do we want this?
        if isinstance(a, IntValue) and isinstance(b, IntValue):
            return as_ufl(a._value / float(b._value))
        
        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            return as_ufl(a._value / b._value)
        
        # construct and initialize a new Division object
        self = AlgebraOperator.__new__(cls)
        self._init(a, b)
        return self
    
    def _init(self, a, b):
        #ufl_assert(isinstance(a, Expr) and isinstance(b, Expr), "Expecting Expr instances.")
        (isinstance(a, Expr) and isinstance(b, Expr)) or error("Expecting Expr instances.")
        self._a = a
        self._b = b
        self._repr = "Division(%r, %r)" % (self._a, self._b)

    def __init__(self, a, b):
        AlgebraOperator.__init__(self)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._a.free_indices()
    
    def index_dimensions(self):
        return self._a.index_dimensions()
    
    def shape(self):
        return self._a.shape()
    
    def evaluate(self, x, mapping, component, index_values):    
        a, b = self.operands()
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        # Avoid integer division
        return float(a) / float(b)
    
    def __str__(self):
        return "(%s / %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return self._repr

class Power(AlgebraOperator):
    __slots__ = ("_a", "_b", "_repr")
    
    def __new__(cls, a, b):
        a = as_ufl(a)
        b = as_ufl(b)
        #ufl_assert(is_true_ufl_scalar(b), "Expecting scalar exponent.")
        #ufl_assert(is_ufl_scalar(b), "Expecting scalar exponent.")
        is_true_ufl_scalar(b) or error("Expecting scalar exponent.")
        is_ufl_scalar(b) or error("Expecting scalar exponent.")
        
        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            return as_ufl(a._value ** b._value)
        if b == 1:
            return a
        if b == 0:
            return IntValue(1)
        
        # construct and initialize a new Power object
        self = AlgebraOperator.__new__(cls)
        self._init(a, b)
        return self
    
    def _init(self, a, b):
        #ufl_assert(isinstance(a, Expr) and isinstance(b, Expr), "Expecting Expr instances.")
        (isinstance(a, Expr) and isinstance(b, Expr)) or error("Expecting Expr instances.")
        self._a = a
        self._b = b
        self._repr = "Power(%r, %r)" % (self._a, self._b)

    def __init__(self, a, b):
        AlgebraOperator.__init__(self)
    
    def operands(self):
        return (self._a, self._b)
    
    def free_indices(self):
        return self._a.free_indices()
    
    def index_dimensions(self):
        return self._a.index_dimensions()
    
    def shape(self):
        return ()
    
    def evaluate(self, x, mapping, component, index_values):    
        a, b = self.operands()
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        return a**b
    
    def __str__(self):
        return "(%s ** %s)" % (str(self._a), str(self._b))
    
    def __repr__(self):
        return self._repr

class Abs(AlgebraOperator):
    __slots__ = ("_a", "_repr")
    
    def __init__(self, a):
        AlgebraOperator.__init__(self)
        ufl_assert(isinstance(a, Expr), "Expecting Expr instance.")
        isinstance(a, Expr) or error("Expecting Expr instances.")
        self._a = a
        self._repr = "Abs(%r)" % self._a
    
    def operands(self):
        return (self._a, )
    
    def free_indices(self):
        return self._a.free_indices()
    
    def index_dimensions(self):
        return self._a.index_dimensions()
    
    def shape(self):
        return self._a.shape()
    
    def evaluate(self, x, mapping, component, index_values):    
        a = self._a.evaluate(x, mapping, component, index_values)
        return abs(a)
    
    def __str__(self):
        return "| %s |" % str(self._a)
    
    def __repr__(self):
        return self._repr
