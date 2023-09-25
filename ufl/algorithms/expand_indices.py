"""This module defines expression transformation utilities.

These utilities are for expanding free indices in expressions to explicit fixed indices only.
"""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009.

from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.classes import Terminal
from ufl.constantvalue import Zero
from ufl.core.multiindex import FixedIndex, Index, MultiIndex
from ufl.differentiation import Grad
from ufl.utils.stacks import Stack, StackDict


class IndexExpander(ReuseTransformer):
    """Index expander."""

    def __init__(self):
        """Initialise."""
        ReuseTransformer.__init__(self)
        self._components = Stack()
        self._index2value = StackDict()

    def component(self):
        """Return current component tuple."""
        if self._components:
            return self._components.peek()
        return ()

    def terminal(self, x):
        """Apply to terminal."""
        if x.ufl_shape:
            c = self.component()
            if len(x.ufl_shape) != len(c):
                raise ValueError("Component size mismatch.")
            return x[c]
        return x

    def form_argument(self, x):
        """Apply to form_argument."""
        sh = x.ufl_shape
        if sh == ():
            return x
        else:
            e = x.ufl_element()
            r = len(sh)

            # Get component
            c = self.component()
            if r != len(c):
                raise ValueError("Component size mismatch.")

            # Map it through an eventual symmetry mapping
            c = min(i for i, j in e.components.items() if j == e.components[c])
            if r != len(c):
                raise ValueError("Component size mismatch after symmetry mapping.")

            return x[c]

    def zero(self, x):
        """Apply to zero."""
        if len(x.ufl_shape) != len(self.component()):
            raise ValueError("Component size mismatch.")

        s = set(x.ufl_free_indices) - set(i.count() for i in self._index2value.keys())
        if s:
            raise ValueError(f"Free index set mismatch, these indices have no value assigned: {s}.")

        # There is no index/shape info in this zero because that is asserted above
        return Zero()

    def scalar_value(self, x):
        """Apply to scalar_value."""
        if len(x.ufl_shape) != len(self.component()):
            self.print_visit_stack()
        if len(x.ufl_shape) != len(self.component()):
            raise ValueError("Component size mismatch.")

        s = set(x.ufl_free_indices) - set(i.count() for i in self._index2value.keys())
        if s:
            raise ValueError(f"Free index set mismatch, these indices have no value assigned: {s}.")

        return x._ufl_class_(x.value())

    def conditional(self, x):
        """Apply to conditional."""
        c, t, f = x.ufl_operands

        # Not accepting nonscalars in condition
        if c.ufl_shape != ():
            raise ValueError("Not expecting tensor in condition.")

        # Conditional may be indexed, push empty component
        self._components.push(())
        c = self.visit(c)
        self._components.pop()

        # Keep possibly non-scalar components for values
        t = self.visit(t)
        f = self.visit(f)

        return self.reuse_if_possible(x, c, t, f)

    def division(self, x):
        """Apply to division."""
        a, b = x.ufl_operands

        # Not accepting nonscalars in division anymore
        if a.ufl_shape != ():
            raise ValueError("Not expecting tensor in division.")
        if self.component() != ():
            raise ValueError("Not expecting component in division.")

        if b.ufl_shape != ():
            raise ValueError("Not expecting division by tensor.")
        a = self.visit(a)

        # self._components.push(())
        b = self.visit(b)
        # self._components.pop()

        return self.reuse_if_possible(x, a, b)

    def index_sum(self, x):
        """Apply to index_sum."""
        ops = []
        summand, multiindex = x.ufl_operands
        index, = multiindex

        # TODO: For the list tensor purging algorithm, do something like:
        # if index not in self._to_expand:
        #     return self.expr(x, *[self.visit(o) for o in x.ufl_operands])

        for value in range(x.dimension()):
            self._index2value.push(index, value)
            ops.append(self.visit(summand))
            self._index2value.pop()
        return sum(ops)

    def _multi_index_values(self, x):
        """Apply to _multi_index_values."""
        comp = []
        for i in x._indices:
            if isinstance(i, FixedIndex):
                comp.append(i._value)
            elif isinstance(i, Index):
                comp.append(self._index2value[i])
        return tuple(comp)

    def multi_index(self, x):
        """Apply to multi_index."""
        comp = self._multi_index_values(x)
        return MultiIndex(tuple(FixedIndex(i) for i in comp))

    def indexed(self, x):
        """Apply to indexed."""
        A, ii = x.ufl_operands

        # Push new component built from index value map
        self._components.push(self._multi_index_values(ii))

        # Hide index values (doing this is not correct behaviour)
        # for i in ii:
        #     if isinstance(i, Index):
        #         self._index2value.push(i, None)

        result = self.visit(A)

        # Un-hide index values
        # for i in ii:
        #     if isinstance(i, Index):
        #         self._index2value.pop()

        # Reset component
        self._components.pop()
        return result

    def component_tensor(self, x):
        """Apply to component_tensor."""
        # This function evaluates the tensor expression
        # with indices equal to the current component tuple
        expression, indices = x.ufl_operands
        if expression.ufl_shape != ():
            raise ValueError("Expecting scalar base expression.")

        # Update index map with component tuple values
        comp = self.component()
        if len(indices) != len(comp):
            raise ValueError("Index/component mismatch.")
        for i, v in zip(indices.indices(), comp):
            self._index2value.push(i, v)
        self._components.push(())

        # Evaluate with these indices
        result = self.visit(expression)

        # Revert index map
        for _ in comp:
            self._index2value.pop()
        self._components.pop()
        return result

    def list_tensor(self, x):
        """Apply to list_tensor."""
        # Pick the right subtensor and subcomponent
        c = self.component()
        c0, c1 = c[0], c[1:]
        op = x.ufl_operands[c0]
        # Evaluate subtensor with this subcomponent
        self._components.push(c1)
        r = self.visit(op)
        self._components.pop()
        return r

    def grad(self, x):
        """Apply to grad."""
        f, = x.ufl_operands
        if not isinstance(f, (Terminal, Grad)):
            raise ValueError("Expecting expand_derivatives to have been applied.")
        # No need to visit child as long as it is on the form [Grad]([Grad](terminal))
        return x[self.component()]


def expand_indices(e):
    """Expand indices."""
    return apply_transformer(e, IndexExpander())
