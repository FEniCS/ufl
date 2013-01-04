"Algorithms for renumbering of counted objects, currently variables and indices."

# Copyright (C) 2008-2013 Martin Sandve Alnes and Anders Logg
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
# First added:  2009-02-22
# Last changed: 2012-04-12

from itertools import izip
from ufl.common import Stack, StackDict
from ufl.log import error
from ufl.expr import Expr
from ufl.indexing import Index, FixedIndex, MultiIndex
from ufl.indexed import Indexed
from ufl.indexsum import IndexSum
from ufl.variable import Label, Variable
from ufl.algorithms.transformer import ReuseTransformer, apply_transformer

class VariableRenumberingTransformer(ReuseTransformer):
    def __init__(self):
        ReuseTransformer.__init__(self)
        self.variable_map = {}

    def variable(self, o):
        e, l = o.operands()
        v = self.variable_map.get(l)
        if v is None:
            e = self.visit(e)
            l2 = Label(len(self.variable_map))
            v = Variable(e, l2)
            self.variable_map[l] = v
        return v

class IndexRenumberingTransformer(VariableRenumberingTransformer):

    def __init__(self):
        VariableRenumberingTransformer.__init__(self)
        self.index_map = {}

    def index_annotated(self, o):
        new_indices = tuple(map(self.index, o.free_indices()))
        return o.reconstruct(new_indices)
    zero = index_annotated
    scalar_value = index_annotated

    def multi_index(self, o):
        new_indices = tuple(map(self.index, o._indices))
        idims = o.index_dimensions()
        new_idims = dict((b, idims[a]) for (a,b) in izip(o._indices, new_indices) if isinstance(a, Index))
        return MultiIndex(new_indices, new_idims)

    def index(self, o):
        if isinstance(o, FixedIndex):
            return o
        c = o._count
        i = self.index_map.get(c)
        if i is None:
            i = Index(len(self.index_map))
            self.index_map[c] = i
        return i

# TODO: Concepts in this implementation can handle unique
#       renumbering of indices used multiple places, like
#           (v[i]*v[i] + u[i]*u[i]) -> (v[i]*v[i] + u[j]*u[j])
#       which could be a useful invariant some other places.
#       However, there are bugs here.
class IndexRenumberingTransformer2(VariableRenumberingTransformer):

    def __init__(self):
        VariableRenumberingTransformer.__init__(self)

        # The number of indices labeled up to now
        self.index_counter = 0

        # A stack of dicts holding an "old Index" -> "new Index"
        # mapping, with "old Index" -> None meaning undefined in
        # the current scope. Indices get defined and numbered
        # in
        self.index_map = StackDict()

        # Current component, a tuple of FixedIndex and Index
        # objects, which are in the new numbering.
        self.components = Stack()
        self.components.push(())

    def new_index(self):
        "Create a new index using our contiguous numbering."
        i = Index(self.index_counter)
        self.index_counter += 1
        #print "::: Making new index", repr(i)
        return i

    def define_new_indices(self, ii):
        #self.define_indices(ii, [self.new_index() for i in ii])
        ni = []
        for i in ii:
            v = self.new_index()
            ni.append(v)
            #print "::: Defining new index", repr(i), "= ", repr(v)
            if self.index_map.get(i) is not None:
                print ";"*80
                print i
                self.print_visit_stack()
                error("Trying to define already defined index!")
            self.index_map.push(i, v)
        return tuple(ni)

    def define_indices(self, ii, values):
        for i, v in izip(ii, values):
            #print "::: Defining index", repr(i), "= ", repr(v)
            if v is None:
                if self.index_map.get(i) is None:
                    print ";"*80
                    print i
                    self.print_visit_stack()
                    error("Trying to undefine index that isn't defined!")
            else:
                if self.index_map.get(i) is not None:
                    print ";"*80
                    print i
                    self.print_visit_stack()
                    error("Trying to define already defined index!")
            self.index_map.push(i, v)

    def revert_indices(self, ii):
        for i in ii:
            j = self.index_map.pop()
            #print "::: Reverting index", repr(i), "(j =", repr(j), ")"

    #    as_tensor(
    #                 u[i,j]
    #              *  v[i]
    #             , j )
    #             [i]
    # *  (
    #       u[i,j]
    #     * (v + w)[j])

    def index(self, o):
        if isinstance(o, FixedIndex):
            return o
        i = self.index_map.get(o)
        if i is None:
            print ";"*80
            print o
            self.print_visit_stack()
            error("Index %s isn't defined!" % repr(o))
        return i

    def multi_index(self, o):
        new_indices = tuple(map(self.index, o._indices))
        idims = o.index_dimensions()
        new_idims = dict((b, idims[a]) for (a,b) in izip(o._indices, new_indices) if isinstance(a, Index))
        return MultiIndex(new_indices, new_idims)

    def index_annotated(self, o):
        new_indices = tuple(map(self.index, o.free_indices()))
        return o.reconstruct(new_indices)
    zero = index_annotated
    scalar_value = index_annotated

    def expr(self, o, *ops):
        r = self.reuse_if_possible(o, *ops)
        c = self.components.peek()
        if c:
            #if isinstance(r, ListTensor):
            #    pass # TODO: If c has FixedIndex objects, extract subtensor, or evt. move this functionality from ListTensor.__getitem__ to Indexed.__new__
            # Take component
            r = Indexed(r, c)
        return r

    def terminal(self, o):
        r = o
        c = self.components.peek()
        if c:
            r = Indexed(r, c)
        return r

    def _spatial_derivative(self, o, *ops):
        r = self.reuse_if_possible(o, *ops)
        return r

    def _sum(self, o, *ops):
        r = self.reuse_if_possible(o, *ops)
        return r

    def indexed(self, f):
        """Binds indices to component, ending their scope as free indices.
        If indices with the same count occur later in a subexpression,
        they represent new indices in a different scope."""

        # Get expression and indices
        g, ii = f.operands()

        # Get values of indices
        c = self.multi_index(ii)

        # Define indices as missing
        jj = [i for i in ii if isinstance(i, Index)]
        jj = tuple(jj)
        #print "::: NOT defining indices as None:", jj
        #self.define_indices(jj, (None,)*len(jj))

        # Push new component
        self.components.push(c)

        # Evaluate expression
        r = self.visit(g)

        # Pop component
        self.components.pop()

        # Revert indices to previous state
        #self.revert_indices(jj)

        return r

    def index_sum(self, o):
        "Defines a new index."
        f, ii = o.operands()
        ni = self.define_new_indices(ii)
        g = self.visit(f)
        r = IndexSum(g, ni)
        self.revert_indices(ii)
        return r

    def component_tensor(self, o):
        """Maps component to indices."""
        f, ii = o.operands()

        # Read component and push new one
        c = self.components.peek()
        self.components.push(())

        # Map component to indices
        self.define_indices(ii, c)

        # Evaluate!
        r = self.visit(f)

        # Pop component to revert to the old
        self.components.pop()

        # Revert index definitions
        self.revert_indices(ii)

        return r

def renumber_indices1(expr):
    if isinstance(expr, Expr) and expr.free_indices():
        error("Not expecting any free indices left in expression.")
    return apply_transformer(expr, IndexRenumberingTransformer())

def renumber_indices2(expr):
    if isinstance(expr, Expr) and expr.free_indices():
        error("Not expecting any free indices left in expression.")
    return apply_transformer(expr, IndexRenumberingTransformer2())

renumber_indices = renumber_indices1

def renumber_variables(expr):
    if isinstance(expr, Expr) and expr.free_indices():
        error("Not expecting any free indices left in expression.")
    return apply_transformer(expr, VariableRenumberingTransformer())

