"Basic algebra operations."

# Copyright (C) 2008-2013 Martin Sandve Alnes
#
# This file is part of UFL.
#
# UFL is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UFL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UFL. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Anders Logg, 2008
#
# First added:  2008-05-20
# Last changed: 2013-01-02

from itertools import chain

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import product, mergedicts, subdict, EmptyDict
from ufl.expr import Expr
from ufl.operatorbase import AlgebraOperator
from ufl.constantvalue import Zero, zero, ScalarValue, IntValue, is_ufl_scalar, is_true_ufl_scalar, as_ufl
from ufl.indexutils import unique_indices
from ufl.sorting import sorted_expr
from ufl.precedence import parstr

#--- Algebraic operators ---

class Sum(AlgebraOperator):
    __slots__ = ("_operands",)

    def __new__(cls, *operands): # TODO: This whole thing seems a bit complicated... Can it be simplified? Maybe we can merge some loops for efficiency?
        ufl_assert(operands, "Can't take sum of nothing.")
        #if not operands:
        #    return Zero() # Allowing this leads to zeros with invalid type information in other places, need indices and shape

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
        if any(sh != o.shape() for o in operands[1:]):
            error("Shape mismatch in Sum.")
        if any((set(fi) ^ set(o.free_indices())) for o in operands[1:]):
            error("Can't add expressions with different free indices.")

        # sort operands in a canonical order
        operands = sorted_expr(operands)

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
        ops = [parstr(o, self) for o in self._operands]
        if False:
            # Implementation with line splitting:
            limit = 70
            delimop = " + \\\n    + "
            op = " + "
            s = ops[0]
            n = len(s)
            for o in ops[1:]:
                m = len(o)
                if n+m > limit:
                    s += delimop
                    n = m
                else:
                    s += op
                    n += m
                s += o
            return s
        # Implementation with no line splitting:
        return "%s" % " + ".join(ops)

    def __repr__(self):
        return "Sum(%s)" % ", ".join(repr(o) for o in self._operands)

class Product(AlgebraOperator):
    """The product of two or more UFL objects."""
    __slots__ = ("_operands", "_free_indices", "_index_dimensions",)

    def __new__(cls, *operands):
        # Make sure everything is an Expr
        operands = [as_ufl(o) for o in operands]

        # Make sure everything is scalar
        #ufl_assert(not any(o.shape() for o in operands),
        #    "Product can only represent products of scalars.")
        if any(o.shape() for o in operands):
            error("Product can only represent products of scalars.")

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
        operands = scalars + sorted_expr(nonscalars)

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
        self._index_dimensions = mergedicts([o.index_dimensions() for o in operands]) or EmptyDict

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
        ops = [parstr(o, self) for o in self._operands]
        if False:
            # Implementation with line splitting:
            limit = 70
            delimop = " * \\\n    * "
            op = " * "
            s = ops[0]
            n = len(s)
            for o in ops[1:]:
                m = len(o)
                if n+m > limit:
                    s += delimop
                    n = m
                else:
                    s += op
                    n += m
                s += o
            return s
        # Implementation with no line splitting:
        return "%s" % " * ".join(ops)

    def __repr__(self):
        return "Product(%s)" % ", ".join(repr(o) for o in self._operands)

class Division(AlgebraOperator):
    __slots__ = ("_a", "_b",)

    def __new__(cls, a, b):
        a = as_ufl(a)
        b = as_ufl(b)

        # Assertions
        # TODO: Enabled workaround for nonscalar division in __div__,
        # so maybe we can keep this assertion. Some algorithms may need updating.
        if not is_ufl_scalar(a):
            error("Expecting scalar nominator in Division.")
        if not is_true_ufl_scalar(b):
            error("Division by non-scalar is undefined.")
        if isinstance(b, Zero):
            error("Division by zero!")

        # Simplification a/b -> a
        if isinstance(a, Zero) or b == 1:
            return a
        # Simplification "literal a / literal b" -> "literal value of a/b"
        # Avoiding integer division by casting to float
        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            return as_ufl(float(a._value) / float(b._value))
        # Simplification "a / a" -> "1"
        if not a.free_indices() and not a.shape() and a == b:
            return as_ufl(1)

        # construct and initialize a new Division object
        self = AlgebraOperator.__new__(cls)
        self._init(a, b)
        return self

    def _init(self, a, b):
        #ufl_assert(isinstance(a, Expr) and isinstance(b, Expr), "Expecting Expr instances.")
        if not (isinstance(a, Expr) and isinstance(b, Expr)):
            error("Expecting Expr instances.")
        self._a = a
        self._b = b

    def __init__(self, a, b):
        AlgebraOperator.__init__(self)

    def operands(self):
        return (self._a, self._b)

    def free_indices(self):
        return self._a.free_indices()

    def index_dimensions(self):
        return self._a.index_dimensions()

    def shape(self):
        return () # self._a.shape()

    def evaluate(self, x, mapping, component, index_values):
        a, b = self.operands()
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        # Avoiding integer division by casting to float
        return float(a) / float(b)

    def __str__(self):
        return "%s / %s" % (parstr(self._a, self), parstr(self._b, self))

    def __repr__(self):
        return "Division(%r, %r)" % (self._a, self._b)

class Power(AlgebraOperator):
    __slots__ = ("_a", "_b",)

    def __new__(cls, a, b):
        a = as_ufl(a)
        b = as_ufl(b)
        if not is_true_ufl_scalar(a): error("Cannot take the power of a non-scalar expression.")
        if not is_true_ufl_scalar(b): error("Cannot raise an expression to a non-scalar power.")

        if isinstance(a, ScalarValue) and isinstance(b, ScalarValue):
            return as_ufl(a._value ** b._value)
        if a == 0 and isinstance(b, ScalarValue):
            bf = float(b)
            if bf < 0:
                error("Division by zero, annot raise 0 to a negative power.")
            else:
                return zero()
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
        if not (isinstance(a, Expr) and isinstance(b, Expr)):
            error("Expecting Expr instances.")
        self._a = a
        self._b = b

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
        return "%s ** %s" % (parstr(self._a, self), parstr(self._b, self))

    def __repr__(self):
        return "Power(%r, %r)" % (self._a, self._b)

class Abs(AlgebraOperator):
    __slots__ = ("_a",)

    def __init__(self, a):
        AlgebraOperator.__init__(self)
        ufl_assert(isinstance(a, Expr), "Expecting Expr instance.")
        if not isinstance(a, Expr): error("Expecting Expr instances.")
        self._a = a

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
        return "| %s |" % parstr(self._a, self)

    def __repr__(self):
        return "Abs(%r)" % self._a
