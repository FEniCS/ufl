"""Classes used to group scalar expressions into expressions with rank > 0."""

# Copyright (C) 2008-2011 Martin Sandve Alnes
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
# First added:  2008-03-31
# Last changed: 2011-04-15

from ufl.log import warning, error
from ufl.common import subdict
from ufl.assertions import ufl_assert
from ufl.expr import Expr, WrapperType
from ufl.constantvalue import as_ufl, Zero
from ufl.indexing import Indexed, Index, FixedIndex, MultiIndex, indices

# --- Classes representing tensors of UFL expressions ---

class ListTensor(WrapperType):
    """UFL operator type: Wraps a list of expressions into a tensor valued expression of one higher rank."""
    __slots__ = ("_expressions", "_free_indices", "_shape", "_repr")

    def __new__(cls, *expressions):
        # All lists and tuples should already be unwrapped in as_tensor
        if any(not isinstance(e, Expr) for e in expressions):
            error("Expecting only UFL expressions in ListTensor constructor.")

        # Get properties of the first expression
        e0 = expressions[0]
        sh    = e0.shape()
        fi    = e0.free_indices()
        idim  = e0.index_dimensions()

        # Obviously, each subextpression must have the same shape
        if any(sh != e.shape() for e in expressions):
            error("ListTensor assumption 1 failed, "\
                  "please report this incident as a potential bug.")

        # Are these assumptions correct? Need to think
        # through the listtensor concept and free indices.
        # Are there any cases where it makes sense to even
        # have any free indices here?
        if any(set(fi) - set(e.free_indices()) for e in expressions):
            error("ListTensor assumption 2 failed, "\
                  "please report this incident as a potential bug.")
        if any(idim != e.index_dimensions() for e in expressions):
            error("ListTensor assumption 3 failed, "\
                  "please report this incident as a potential bug.")

        # Simplify to Zero if possible
        if all(isinstance(e, Zero) for e in expressions):
            shape = (len(expressions),) + sh
            return Zero(shape, fi, idim)

        return WrapperType.__new__(cls)

    def __init__(self, *expressions):
        WrapperType.__init__(self)
        e0 = expressions[0]
        sh = e0.shape()
        self._shape = (len(expressions),) + sh
        self._expressions = tuple(expressions)

        indexset = set(e0.free_indices())
        ufl_assert(all(not (indexset ^ set(e.free_indices())) for e in self._expressions),\
            "Can't combine subtensor expressions with different sets of free indices.")

        self._repr = "ListTensor(%s)" % ", ".join(repr(e) for e in self._expressions)

    def operands(self):
        return self._expressions

    def free_indices(self):
        return self._expressions[0].free_indices()

    def index_dimensions(self):
        return self._expressions[0].index_dimensions()

    def shape(self):
        return self._shape

    def evaluate(self, x, mapping, component, index_values):
        ufl_assert(len(component) == len(self._shape),
                   "Can only evaluate scalars, expecting a component "\
                   "tuple of length %d, not %s." % (len(self._shape), component))
        a = self._expressions[component[0]]
        component = component[1:]
        a = a.evaluate(x, mapping, component, index_values)
        return a

    def __getitem__(self, key):
        origkey = key

        if isinstance(key, MultiIndex):
            key = key._indices
        if not isinstance(key, tuple):
            key = (key,)
        k = key[0]
        if isinstance(k, (int, FixedIndex)):
            sub = self._expressions[int(k)]
            return sub if len(key) == 1 else sub[key[1:]]

        return Expr.__getitem__(self, origkey)
    
    def __str__(self):
        def substring(expressions, indent):
            ind = " "*indent
            if any(isinstance(e, ListTensor) for e in expressions):
                substrings = []
                for e in expressions:
                    if isinstance(e, ListTensor):
                        substrings.append(substring(e._expressions, indent+2))
                    else:
                        substrings.append(str(e))
                s = (",\n" + ind).join(substrings)
                return "%s[\n%s%s\n%s]" % (ind, ind, s, ind)
            else:
                s = ", ".join(str(e) for e in expressions)
                return "%s[%s]" % (ind, s)
        return substring(self._expressions, 0)

    def __repr__(self):
        return self._repr

class ComponentTensor(WrapperType):
    """UFL operator type: Maps the free indices of a scalar valued expression to tensor axes."""
    __slots__ = ("_expression", "_indices", "_free_indices",
                 "_index_dimensions", "_shape", "_str", "_repr")

    def __new__(cls, expression, indices):
        if isinstance(expression, Zero):
            if isinstance(indices, MultiIndex):
                indices = tuple(indices)
            elif not isinstance(indices, tuple):
                indices = (indices,)
            dims = expression.index_dimensions()
            shape = tuple(dims[i] for i in indices)
            fi = tuple(set(expression.free_indices()) - set(indices))
            idim = dict((i, dims[i]) for i in fi)
            return Zero(shape, fi, idim)
        return WrapperType.__new__(cls)

    def __init__(self, expression, indices):
        WrapperType.__init__(self)
        ufl_assert(isinstance(expression, Expr), "Expecting ufl expression.")
        ufl_assert(expression.shape() == (), "Expecting scalar valued expression.")
        self._expression = expression
        
        ufl_assert(all(isinstance(i, Index) for i in indices),
           "Expecting sequence of Index objects, not %s." % repr(indices))
        
        dims = expression.index_dimensions()
        
        if not isinstance(indices, MultiIndex): # if constructed from repr
            indices = MultiIndex(indices, subdict(dims, indices))
        self._indices = indices
        
        eset = set(expression.free_indices())
        iset = set(self._indices)
        freeset = eset - iset
        self._free_indices = tuple(freeset)

        missingset = iset - eset
        if missingset:
            error("Missing indices %s in expression %s." % (missingset, expression))
        
        self._index_dimensions = dict((i, dims[i]) for i in self._free_indices)
        
        self._shape = tuple(dims[i] for i in self._indices)

        self._str = "{ A | A_{%s} = %s }" % (self._indices, self._expression)
        self._repr = "ComponentTensor(%r, %r)" % (self._expression, self._indices)
    
    def operands(self):
        return (self._expression, self._indices)
    
    def free_indices(self):
        return self._free_indices
    
    def index_dimensions(self):
        return self._index_dimensions
    
    def shape(self):
        return self._shape
    
    def evaluate(self, x, mapping, component, index_values):
        indices = self._indices
        a = self._expression
        
        # Map component to indices
        for i, c in zip(indices, component):
            index_values.push(i, c)
        
        a = a.evaluate(x, mapping, (), index_values)
        
        for _ in component:
            index_values.pop()
        
        return a

    def __str__(self):
        return self._str
    
    def __repr__(self):
        return self._repr

# --- User-level functions to wrap expressions in the correct way ---

def numpy2nestedlists(arr):
    from numpy import ndarray
    if not isinstance(arr, ndarray):
        return arr
    return [numpy2nestedlists(arr[k]) for k in range(arr.shape[0])]

def tmp():
    # FIXME: Should have no list or tuple here, convert those in as_tensor!
    if isinstance(expressions[0], (list, tuple)):
        ufl_assert(all(isinstance(e, (list, tuple)) for e in expressions),
                   "Expecting all or no arguments being lists or tuples.")
        expressions = [ListTensor(*sub) for sub in expressions]

    if not all(isinstance(e, ListTensor) for e in expressions):
        expressions = [as_ufl(e) for e in expressions]
        ufl_assert(all(isinstance(e, Expr) for e in expressions), \
                   "Expecting list of subtensors or expressions.")

def _as_list_tensor(expressions):
    if isinstance(expressions, (list, tuple)):
        expressions = [_as_list_tensor(e) for e in expressions] 
        return ListTensor(*expressions)
    else:
        return as_ufl(expressions)

def from_numpy_to_lists(expressions):
    try:
        import numpy
        if isinstance(expressions, numpy.ndarray):
            expressions = numpy2nestedlists(expressions)
    except:
        pass
    return expressions

def as_tensor(expressions, indices = None):
    """UFL operator: Make a tensor valued expression.

    This works in two different ways, by using indices or lists.

    1) Returns A such that A[indices] = expressions.
    If indices are provided, expressions must be a scalar
    valued expression with all the provided indices among
    its free indices. This operator will then map each of these
    indices to a tensor axis, thereby making a tensor valued
    expression from a scalar valued expression with free indices.

    2) Returns A such that A[k,...] = expressions[k].
    If no indices are provided, expressions must be a list
    or tuple of expressions. The expressions can also consist
    of recursively nested lists to build higher rank tensors.
    """
    if indices is None:
        # Support numpy array, but avoid importing numpy if not needed
        if not isinstance(expressions, (list, tuple)):
            expressions = from_numpy_to_lists(expressions)

        # Sanity check
        if not isinstance(expressions, (list, tuple)):
            error("Expecting nested list or tuple.")

        # Recursive conversion from nested lists to nested ListTensor objects
        return _as_list_tensor(expressions)
    else:
        # Make sure we have a tuple of indices
        if isinstance(indices, list):
            indices = tuple(indices)
        elif not isinstance(indices, tuple):
            indices = (indices,)

        # Special case for as_tensor(expr, ii) with ii = ()
        if indices == ():
            return expressions

        # Special case for simplification as_tensor(A[ii], ii) -> A
        if isinstance(expressions, Indexed):
            A, ii = expressions.operands()
            if indices == ii._indices:
                return A

        # Make a tensor from given scalar expression with free indices
        return ComponentTensor(expressions, indices)

def as_matrix(expressions, indices = None):
    "UFL operator: As as_tensor(), but limited to rank 2 tensors."
    if indices is None:
        # To avoid importing numpy unneeded, it's quite slow...
        if not isinstance(expressions, (list, tuple)):
            try:
                import numpy
                if isinstance(expressions, numpy.ndarray):
                    expressions = numpy2nestedlists(expressions)
            except:
                pass
        ufl_assert(isinstance(expressions, (list, tuple)),
            "Expecting nested list or tuple of Exprs.")
        ufl_assert(isinstance(expressions[0], (list, tuple)),
            "Expecting nested list or tuple of Exprs.")
        return as_tensor(expressions)

    ufl_assert(len(indices) == 2, "Expecting exactly two indices.")
    return as_tensor(expressions, indices)

def as_vector(expressions, index = None):
    "UFL operator: As as_tensor(), but limited to rank 1 tensors."
    if index is not None:
        ufl_assert(isinstance(index, Index), "Expecting Index object.")
        index = (index,)
    return as_tensor(expressions, index)

def as_scalar(expression):
    """Given a scalar or tensor valued expression A,
    returns either the tuple
      (a,b) = (A, ())
    or
      (a,b) = (A[indices], indices)
    such that a is always a scalar valued expression."""
    ii = indices(expression.rank())
    if ii:
        expression = expression[ii]
    return expression, ii

def relabel(A, indexmap):
    "UFL operator: Relabel free indices of A with new indices, using the given mapping."
    ii = tuple(sorted(indexmap.keys()))
    jj = tuple(indexmap[i] for i in ii)
    ufl_assert(all(isinstance(i, Index) for i in ii), "Expecting Index objects.")
    ufl_assert(all(isinstance(j, Index) for j in jj), "Expecting Index objects.")
    return as_tensor(A, ii)[jj]

# --- Experimental support for dyadic notation:

def unit_list(i, n):
    return [(1 if i == j else 0) for j in xrange(n)]

def unit_list2(i, j, n):
    return [[(1 if (i == i0 and j == j0) else 0) for j0 in xrange(n)] for i0 in xrange(n)]

def unit_vector(i, d):
    "UFL value: A constant unit vector in direction i with dimension d."
    return as_vector(unit_list(i, d))

def unit_vectors(d):
    "UFL value: A tuple of constant unit vectors in all directions with dimension d."
    return tuple(unit_vector(i, d) for i in range(d))

def unit_matrix(i, j, d):
    "UFL value: A constant unit matrix in direction i,j with dimension d."
    return as_matrix(unit_list2(i, j, d))

def unit_matrices(d):
    "UFL value: A tuple of constant unit matrices in all directions with dimension d."
    return tuple(unit_matrix(i, j, d) for i in range(d) for j in range(d))

def dyad(d, *iota):
    "TODO: Develop this concept, can e.g. write A[i,j]*dyad(j,i) for the transpose."
    I = Identity(d)
    i = iota[0]
    e = as_vector(I[i,:], i)
    for i in iota[1:]:
        e = outer(e, as_vector(I[i,:], i))
    return e

def _test():
    #from ufl.tensors import unit_vector, unit_vectors, unit_matrix, unit_matrices
    from ufl.objects import triangle
    cell = triangle
    d = cell.geometric_dimension()
    ei, ej, ek = unit_vectors(d)
    eii, eij, eik, eji, ejj, ejk, eki, ekj, ekk = unit_matrices(d)
    print ei
    print ej
    print ek
    print eii
    print eij
    print eik
    print eji
    print ejj
    print ejk
    print eki
    print ekj
    print ekk

