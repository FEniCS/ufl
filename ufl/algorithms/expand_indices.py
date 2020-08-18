# -*- coding: utf-8 -*-
"""This module defines expression transformation utilities,
for expanding free indices in expressions to explicit fixed
indices only."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2009.

from ufl.log import error
from ufl.utils.stacks import Stack, StackDict
from ufl.classes import Terminal, ListTensor
from ufl.constantvalue import Zero
from ufl.core.multiindex import Index, FixedIndex, MultiIndex
from ufl.differentiation import Grad
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer
from ufl.corealg.traversal import unique_pre_traversal


class IndexExpander(ReuseTransformer):
    """..."""

    def __init__(self):
        ReuseTransformer.__init__(self)
        self._components = Stack()
        self._index2value = StackDict()

    def component(self):
        "Return current component tuple."
        if self._components:
            return self._components.peek()
        return ()

    def terminal(self, x):
        if x.ufl_shape:
            c = self.component()
            if len(x.ufl_shape) != len(c):
                error("Component size mismatch.")
            return x[c]
        return x

    def form_argument(self, x):
        sh = x.ufl_shape
        if sh == ():
            return x
        else:
            e = x.ufl_element()
            r = len(sh)

            # Get component
            c = self.component()
            if r != len(c):
                error("Component size mismatch.")

            # Map it through an eventual symmetry mapping
            s = e.symmetry()
            c = s.get(c, c)
            if r != len(c):
                error("Component size mismatch after symmetry mapping.")

            return x[c]

    def zero(self, x):
        if len(x.ufl_shape) != len(self.component()):
            error("Component size mismatch.")

        s = set(x.ufl_free_indices) - set(i.count() for i in self._index2value.keys())
        if s:
            error("Free index set mismatch, these indices have no value assigned: %s." % str(s))

        # There is no index/shape info in this zero because that is asserted above
        return Zero()

    def scalar_value(self, x):
        if len(x.ufl_shape) != len(self.component()):
            self.print_visit_stack()
        if len(x.ufl_shape) != len(self.component()):
            error("Component size mismatch.")

        s = set(x.ufl_free_indices) - set(i.count() for i in self._index2value.keys())
        if s:
            error("Free index set mismatch, these indices have no value assigned: %s." % str(s))

        return x._ufl_class_(x.value())

    def conditional(self, x):
        c, t, f = x.ufl_operands

        # Not accepting nonscalars in condition
        if c.ufl_shape != ():
            error("Not expecting tensor in condition.")

        # Conditional may be indexed, push empty component
        self._components.push(())
        c = self.visit(c)
        self._components.pop()

        # Keep possibly non-scalar components for values
        t = self.visit(t)
        f = self.visit(f)

        return self.reuse_if_possible(x, c, t, f)

    def division(self, x):
        a, b = x.ufl_operands

        # Not accepting nonscalars in division anymore
        if a.ufl_shape != ():
            error("Not expecting tensor in division.")
        if self.component() != ():
            error("Not expecting component in division.")

        if b.ufl_shape != ():
            error("Not expecting division by tensor.")
        a = self.visit(a)

        # self._components.push(())
        b = self.visit(b)
        # self._components.pop()

        return self.reuse_if_possible(x, a, b)

    def index_sum(self, x):
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
        comp = []
        for i in x._indices:
            if isinstance(i, FixedIndex):
                comp.append(i._value)
            elif isinstance(i, Index):
                comp.append(self._index2value[i])
        return tuple(comp)

    def multi_index(self, x):
        comp = self._multi_index_values(x)
        return MultiIndex(tuple(FixedIndex(i) for i in comp))

    def indexed(self, x):
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
        # This function evaluates the tensor expression
        # with indices equal to the current component tuple
        expression, indices = x.ufl_operands
        if expression.ufl_shape != ():
            error("Expecting scalar base expression.")

        # Update index map with component tuple values
        comp = self.component()
        if len(indices) != len(comp):
            error("Index/component mismatch.")
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
        f, = x.ufl_operands
        if not isinstance(f, (Terminal, Grad)):
            error("Expecting expand_derivatives to have been applied.")
        # No need to visit child as long as it is on the form [Grad]([Grad](terminal))
        return x[self.component()]


def expand_indices(e):
    return apply_transformer(e, IndexExpander())


def purge_list_tensors(expr):
    """Get rid of all ListTensor instances by expanding
    expressions to use their components directly.
    Will usually increase the size of the expression."""
    if any(isinstance(subexpr, ListTensor) for subexpr in unique_pre_traversal(expr)):
        return expand_indices(expr)  # TODO: Only expand what's necessary to get rid of list tensors
    return expr
