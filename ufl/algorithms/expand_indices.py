"""This module defines expression transformation utilities,
for expanding free indices in expressions to explicit fixed
indices only."""

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
# Modified by Anders Logg, 2009.
#
# First added:  2008-04-19
# Last changed: 2012-04-12

from itertools import izip
from ufl.log import error
from ufl.common import Stack, StackDict
from ufl.assertions import ufl_assert
from ufl.finiteelement import TensorElement
from ufl.classes import Expr, Terminal, ListTensor, IndexSum, Indexed, FormArgument
from ufl.tensors import as_tensor, ComponentTensor
from ufl.permutation import compute_indices
from ufl.constantvalue import Zero
from ufl.indexing import Index, FixedIndex, MultiIndex
from ufl.differentiation import Grad
from ufl.algorithms.graph import Graph
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer, transform_integrands
from ufl.algorithms.analysis import has_type

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
        if x.shape():
            c = self.component()
            ufl_assert(len(x.shape()) == len(c), "Component size mismatch.")
            return x[c]
        return x

    def form_argument(self, x):
        sh = x.shape()
        if sh == ():
            return x
        else:
            e = x.element()
            r = len(sh)

            # Get component
            c = self.component()
            ufl_assert(r == len(c), "Component size mismatch.")

            # Map it through an eventual symmetry mapping
            s = e.symmetry()
            c = s.get(c, c)
            ufl_assert(r == len(c), "Component size mismatch after symmetry mapping.")

            return x[c]

    def zero(self, x):
        ufl_assert(len(x.shape()) == len(self.component()), "Component size mismatch.")
        s = set(x.free_indices()) - set(self._index2value.keys())
        if s: error("Free index set mismatch, these indices have no value assigned: %s." % str(s))
        # There is no index/shape info in this zero because that is asserted above
        return Zero()

    def scalar_value(self, x):
        if len(x.shape()) != len(self.component()):
            self.print_visit_stack()
        ufl_assert(len(x.shape()) == len(self.component()), "Component size mismatch.")

        s = set(x.free_indices()) - set(self._index2value.keys())
        if s: error("Free index set mismatch, these indices have no value assigned: %s." % str(s))

        return x._uflclass(x.value())

    def division(self, x):
        a, b = x.operands()

        # Not accepting nonscalars in division anymore
        ufl_assert(a.shape() == (), "Not expecting tensor in division.")
        ufl_assert(self.component() == (), "Not expecting component in division.")

        ufl_assert(b.shape() == (), "Not expecting division by tensor.")
        a = self.visit(a)

        #self._components.push(())
        b = self.visit(b)
        #self._components.pop()

        return self.reuse_if_possible(x, a, b)

    def index_sum(self, x):
        ops = []
        summand, multiindex = x.operands()
        index, = multiindex

        # TODO: For the list tensor purging algorithm, do something like:
        # if index not in self._to_expand:
        #     return self.expr(x, *[self.visit(o) for o in x.operands()])

        for value in range(x.dimension()):
            self._index2value.push(index, value)
            ops.append(self.visit(summand))
            self._index2value.pop()
        return sum(ops)

    def _multi_index(self, x):
        comp = []
        for i in x:
            if isinstance(i, FixedIndex):
                comp.append(i._value)
            elif isinstance(i, Index):
                comp.append(self._index2value[i])
        return tuple(comp)

    def multi_index(self, x):
        return MultiIndex(self._multi_index(x), {})

    def indexed(self, x):
        A, ii = x.operands()

        # Push new component built from index value map
        self._components.push(self._multi_index(ii))

        # Hide index values (doing this is not correct behaviour)
        #for i in ii:
        #    if isinstance(i, Index):
        #        self._index2value.push(i, None)

        result = self.visit(A)

        # Un-hide index values
        #for i in ii:
        #    if isinstance(i, Index):
        #        self._index2value.pop()

        # Reset component
        self._components.pop()
        return result

    def component_tensor(self, x):
        # This function evaluates the tensor expression
        # with indices equal to the current component tuple
        expression, indices = x.operands()
        ufl_assert(expression.shape() == (), "Expecting scalar base expression.")

        # Update index map with component tuple values
        comp = self.component()
        ufl_assert(len(indices) == len(comp), "Index/component mismatch.")
        for i, v in izip(indices._indices, comp):
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
        op = x.operands()[c0]
        # Evaluate subtensor with this subcomponent
        self._components.push(c1)
        r = self.visit(op)
        self._components.pop()
        return r

    def grad(self, x):
        f, = x.operands()
        ufl_assert(isinstance(f, (Terminal, Grad)),
                   "Expecting expand_derivatives to have been applied.")
        # No need to visit child as long as it is on the form [Grad]([Grad](terminal))
        return x[self.component()]

def expand_indices(e):
    return apply_transformer(e, IndexExpander())

def purge_list_tensors(e):
    """Get rid of all ListTensor instances by expanding
    expressions to use their components directly.
    Will usually increase the size of the expression."""
    if has_type(e, ListTensor):
        return expand_indices(e) # TODO: Only expand what's necessary to get rid of list tensors
    return e
