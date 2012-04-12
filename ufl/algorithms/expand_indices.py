"""This module defines expression transformation utilities,
for expanding free indices in expressions to explicit fixed
indices only."""

# Copyright (C) 2008-2012 Martin Sandve Alnes
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
from ufl.differentiation import SpatialDerivative
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
        return Zero() # TODO: Don't remember when reading this code: is it right that there is no index/shape info in this zero?

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

    def spatial_derivative(self, x):
        f, ii = x.operands()
        ufl_assert(isinstance(f, (Terminal, SpatialDerivative, Indexed, ListTensor, ComponentTensor)),
                   "Expecting expand_derivatives to have been applied.")

        # Taking component if necessary
        fold = f
        f = self.visit(f)

        #ii = self.visit(ii) # mapping to constant if necessary

        # Map free index to a value
        iiold = ii
        i, = ii
        if isinstance(i, Index):
            ii = MultiIndex((FixedIndex(self._index2value[i]),), {})

        # Hide used index i (doing this is not correct behaviour)
        #if isinstance(i, Index):
        #    self._index2value.push(i, None)
        #    pushed = True
        #else:
        #    pushed = False

        #self.reuse_if_possible(x, f, ii)
        if f == fold and ii == iiold:
            result = x
        else:
            result = f.dx(*ii)

        # Unhide used index i
        #if pushed:
        #    self._index2value.pop()

        return result

def expand_indices(e):
    return apply_transformer(e, IndexExpander())

def expand_indices2(e):
    return transform_integrands(e, expand_indices2_alg)

def expand_indices2_alg(e):
    assert isinstance(e, Expr)

    G = Graph(e)
    V, E = G
    n = len(V)
    Vout = G.Vout()
    Vin = G.Vin()

    # Cache free indices
    fi   = []
    idim = []
    for i, v in enumerate(V):
        if isinstance(v, MultiIndex):
            #ii = tuple(j for j in v._indices if isinstance(j, Index))
            #idims = {} # Hard problem: Need index dimensions but they're defined by the parent.
            ii = v.free_indices()
            try:
                idims = v.index_dimensions()
            except:
                print "The type in question is", type(v)
                print str(v)
                print repr(v)
                print "expression type: ", type(e)
                print "parent types: ", [type(V[j]) for j in Vin[i]]
                print "parents children: ", [type(V[k]) for j in Vin[i] for k in Vout[j]]
                idims = {}
                raise
        else:
            try:
                ii = v.free_indices()
                idims = v.index_dimensions()
            except:
                ii = ()
                idims = {}
        fi.append(ii)
        idim.append(idims)

    # Cache of expanded expressions
    V2 = [{} for _ in V]
    def getv(i, indmap):
        return V2[i][tuple(indmap[j] for j in fi[i])]

    # Reversed enumeration list
    RV = list(enumerate(V))
    RV.reverse()

    # Map of current index values
    indmap = StackDict()

    # Expand all vertices in turn
    for i, v in enumerate(V):
        ii = fi[i]
        if ii:
            idims = idim[i]
            dii = tuple(idims[j] for j in ii)
            perms = compute_indices(dii)
        else:
            perms = [()]

        for p in perms:
            # Map indices to permutation p
            for (j, d) in izip(ii, p):
                assert isinstance(j, Index)
                assert isinstance(d, int)
                indmap.push(j, d)

            if isinstance(v, MultiIndex):
                # Map to FixedIndex tuple
                comp = []
                k = 0
                for j in v._indices:
                    if isinstance(j, FixedIndex):
                        comp.append(j)
                    elif isinstance(j, Index):
                        comp.append(FixedIndex(p[k]))
                        k += 1
                e = MultiIndex(tuple(comp), {})

            elif isinstance(v, IndexSum):
                e = None
                for k in range(v.dimension()):
                    indmap.push(v.index(), k)
                    # Get operands evaluated for this index configuration
                    ops = [getv(j, indmap) for j in Vout[i]]
                    # It is possible to save memory
                    # by reusing some expressions here
                    if e is None:
                        e = ops[0]
                    else:
                        e += ops[0]
                    indmap.pop()

            elif isinstance(v, Indexed):
                # Get operands evaluated for this index configuration
                ops = [getv(j, indmap) for j in Vout[i]]
                A = ops[0]
                comp = ops[1]._indices
                comp = tuple(int(c) for c in comp) # need ints for symmetry mapping
                if isinstance(A, FormArgument) and A.shape():
                    # Get symmetry mapping if any
                    e = A.element()
                    s = None
                    if isinstance(e, TensorElement):
                        s = e.symmetry()
                    if s is None:
                        s = {}
                    # Map component throught the symmetry mapping
                    c = comp
                    ufl_assert(len(A.shape()) == len(c), "Component size mismatch.")
                    comp = s.get(c, c)
                    ufl_assert(len(c) == len(comp), "Component size mismatch after symmetry mapping.")

                e = A[comp]
                if isinstance(A, ListTensor) and isinstance(e, Indexed) and not isinstance(e.operands()[0], (Terminal, SpatialDerivative)):
                    print "="*80
                    print "This is a case where expand_indices2 doesn't work as it should:" # TODO: Must fix before we can employ and optimize this!
                    print str(A)
                    print repr(comp)
                    print str(e)
                    print "="*80
                    import sys
                    sys.exit(-1)

            elif isinstance(v, ComponentTensor):
                import numpy
                A = numpy.ndarray(shape=v.shape(), dtype=object)
                iota = v.operands()[1] #getv(Vout[i][1], indmap) # not to be mapped here
                for k in compute_indices(v.shape()):
                    for (j, d) in izip(iota, k):
                        indmap.push(j, d)
                    Ak = getv(Vout[i][0], indmap)
                    for _ in iota:
                        indmap.pop()
                    A[k] = Ak
                e = as_tensor(A)

            elif isinstance(v, ListTensor):
                ops = [getv(j, indmap) for j in Vout[i]]
                e = v.reconstruct(*ops)

            elif isinstance(v, Terminal):
                # Simply reuse
                e = v

            else:
                # Get operands evaluated for this index configuration
                ops = [getv(j, indmap) for j in Vout[i]]
                # It is possible to save memory
                # by reusing some expressions here
                e = v.reconstruct(*ops)

            # Undo mapping of indices to permutation p
            for _ in p:
                indmap.pop()

            V2[i][p] = e

    return V2[-1][()]

def purge_list_tensors(e):
    """Get rid of all ListTensor instances by expanding
    expressions to use their components directly.
    Will usually increase the size of the expression."""
    if has_type(e, ListTensor):
        return expand_indices(e) # TODO: Only expand what's necessary to get rid of list tensors
    return e
