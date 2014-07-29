"Basic algebra operations."

# Copyright (C) 2008-2014 Martin Sandve Alnes
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

from itertools import chain
from six import iteritems

from ufl.log import error, warning
from ufl.assertions import ufl_assert
from ufl.common import product, mergedicts2, subdict, EmptyDict
from ufl.expr import Expr
from ufl.operatorbase import AlgebraOperator
from ufl.constantvalue import Zero, zero, ScalarValue, IntValue, as_ufl
from ufl.checks import is_ufl_scalar, is_true_ufl_scalar
from ufl.indexutils import unique_indices
from ufl.sorting import sorted_expr
from ufl.precedence import parstr
from ufl.core.ufl_type import ufl_type

#--- Algebraic operators ---

@ufl_type(num_ops=2)
class Sum(AlgebraOperator):
    __slots__ = ()

    def __new__(cls, a, b):
        # Make sure everything is an Expr
        a = as_ufl(a)
        b = as_ufl(b)

        # Assert consistent tensor properties
        sh = a.shape()
        fi = a.free_indices()
        fid = a.index_dimensions()
        if b.shape() != sh:
            error("Can't add expressions with different shapes.")
        if set(fi) ^ set(b.free_indices()):
            error("Can't add expressions with different free indices.")

        # Skip adding zero
        if isinstance(a, Zero):
            return b
        elif isinstance(b, Zero):
            return a

        # Handle scalars specially and sort operands
        sa = isinstance(a, ScalarValue)
        sb = isinstance(b, ScalarValue)
        if sa and sb:
            # Apply constant propagation
            return as_ufl(a._value + b._value)
        elif sa:
            # Place scalar first
            #operands = (a, b)
            pass #a, b = a, b
        elif sb:
            # Place scalar first
            #operands = (b, a)
            a, b = b, a
        elif a == b:
            # Replace a+b with 2*foo
            return 2*a
        else:
            # Otherwise sort operands in a canonical order
            #operands = (b, a)
            a, b = sorted_expr((a, b))

        # construct and initialize a new Sum object
        self = AlgebraOperator.__new__(cls)
        self._init(a, b)
        return self

    def _init(self, a, b):
        self.ufl_operands = (a, b)

    def __init__(self, a, b):
        AlgebraOperator.__init__(self)

    def free_indices(self):
        return self.ufl_operands[0].free_indices()

    def index_dimensions(self):
        return self.ufl_operands[0].index_dimensions()

    def shape(self):
        return self.ufl_operands[0].shape()

    def evaluate(self, x, mapping, component, index_values):
        return sum(o.evaluate(x, mapping, component, index_values) for o in self.ufl_operands)

    def __str__(self):
        ops = [parstr(o, self) for o in self.ufl_operands]
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
        return "Sum(%s)" % ", ".join(repr(o) for o in self.ufl_operands)

@ufl_type(num_ops=2)
class Product(AlgebraOperator):
    """The product of two or more UFL objects."""
    __slots__ = ("_free_indices", "_index_dimensions",)

    def __new__(cls, a, b):
        # Make sure everything is an Expr
        a = as_ufl(a)
        b = as_ufl(b)
        operands = (a, b) # TODO: Temporary, rewrite below code to use a,b

        # Make sure everything is scalar
        if a.shape() or b.shape():
            error("Product can only represent products of scalars.")

        # Got any zeros? Return zero.
        if isinstance(a, Zero) or isinstance(b, Zero):
            free_indices     = unique_indices(tuple(chain(a.free_indices(), b.free_indices())))
            index_dimensions = subdict(mergedicts2(a.index_dimensions(), b.index_dimensions()), free_indices)
            return Zero((), free_indices, index_dimensions)

        # Merge if both are scalars
        sa = isinstance(a, ScalarValue)
        sb = isinstance(b, ScalarValue)
        if sa and sb:
            # FIXME: Handle free indices like with zero? I think IntValue may be index annotated now?
            return as_ufl(a._value * b._value)
        elif sa:
            if a._value == 1:
                return b
            # a, b = a, b
        elif sb:
            if b._value == 1:
                return a
            a, b = b, a
        elif a == b:
            # Replace a*a with a**2 # TODO: Why? Maybe just remove this?
            if not a.free_indices():
                return a**2
        else:
            # Sort operands in a canonical order (NB! This is fragile! Small changes here can have large effects.)
            a, b = sorted_expr((a, b))

        # Construct and initialize a new Product object
        self = AlgebraOperator.__new__(cls)
        self._init(a, b)
        return self

    def _init(self, a, b):
        "Constructor, called by __new__ with already checked arguments."
        # Store basic properties
        self.ufl_operands = (a, b)

        # Extract indices
        self._free_indices     = unique_indices(tuple(chain(a.free_indices(), b.free_indices())))
        #self._index_dimensions = frozendict(chain(iteritems(o.index_dimensions()) for o in (a,b))) or EmptyDict
        self._index_dimensions = mergedicts2(a.index_dimensions(), b.index_dimensions()) or EmptyDict

    def __init__(self, a, b):
        AlgebraOperator.__init__(self)

    def free_indices(self):
        return self._free_indices

    def index_dimensions(self):
        return self._index_dimensions

    def shape(self):
        return ()

    def evaluate(self, x, mapping, component, index_values):
        ops = self.ufl_operands
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
        ops = [parstr(o, self) for o in self.ufl_operands]
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
        return "Product(%s)" % ", ".join(repr(o) for o in self.ufl_operands)

@ufl_type(num_ops=2)
class Division(AlgebraOperator):
    __slots__ = ()

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
        self.ufl_operands = (a, b)

    def __init__(self, a, b):
        AlgebraOperator.__init__(self)

    def free_indices(self):
        return self.ufl_operands[0].free_indices()

    def index_dimensions(self):
        return self.ufl_operands[0].index_dimensions()

    def shape(self):
        return () # self.ufl_operands[0].shape()

    def evaluate(self, x, mapping, component, index_values):
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        # Avoiding integer division by casting to float
        return float(a) / float(b)

    def __str__(self):
        return "%s / %s" % (parstr(self.ufl_operands[0], self), parstr(self.ufl_operands[1], self))

    def __repr__(self):
        return "Division(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

@ufl_type(num_ops=2)
class Power(AlgebraOperator):
    __slots__ = ()

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
        self.ufl_operands = (a, b)

    def __init__(self, a, b):
        AlgebraOperator.__init__(self)

    def free_indices(self):
        return self.ufl_operands[0].free_indices()

    def index_dimensions(self):
        return self.ufl_operands[0].index_dimensions()

    def shape(self):
        return ()

    def evaluate(self, x, mapping, component, index_values):
        a, b = self.ufl_operands
        a = a.evaluate(x, mapping, component, index_values)
        b = b.evaluate(x, mapping, component, index_values)
        return a**b

    def __str__(self):
        return "%s ** %s" % (parstr(self.ufl_operands[0], self), parstr(self.ufl_operands[1], self))

    def __repr__(self):
        return "Power(%r, %r)" % (self.ufl_operands[0], self.ufl_operands[1])

@ufl_type(num_ops=1)
class Abs(AlgebraOperator):
    __slots__ = ()

    def __init__(self, a):
        AlgebraOperator.__init__(self, (a,))
        ufl_assert(isinstance(a, Expr), "Expecting Expr instance.")
        if not isinstance(a, Expr): error("Expecting Expr instances.")

    def free_indices(self):
        return self.ufl_operands[0].free_indices()

    def index_dimensions(self):
        return self.ufl_operands[0].index_dimensions()

    def shape(self):
        return self.ufl_operands[0].shape()

    def evaluate(self, x, mapping, component, index_values):
        a = self.ufl_operands[0].evaluate(x, mapping, component, index_values)
        return abs(a)

    def __str__(self):
        return "| %s |" % parstr(self.ufl_operands[0], self)

    def __repr__(self):
        return "Abs(%r)" % self.ufl_operands[0]
