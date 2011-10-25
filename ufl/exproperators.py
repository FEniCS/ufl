"""This module attaches special functions to Expr.
This way we avoid circular dependencies between e.g.
Sum and its superclass Expr."""

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
# First added:  2008-08-18
# Last changed: 2011-04-12

from itertools import chain

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.common import mergedicts, subdict, StackDict
from ufl.expr import Expr
from ufl.constantvalue import Zero, ScalarValue, FloatValue, IntValue, is_python_scalar, is_true_ufl_scalar, as_ufl, python_scalar_types
from ufl.algebra import Sum, Product, Division, Power, Abs
from ufl.tensoralgebra import Transposed, Dot
from ufl.indexing import IndexBase, FixedIndex, Index, Indexed, IndexSum, indices
from ufl.indexutils import repeated_indices, unique_indices, single_indices
from ufl.tensors import as_tensor
from ufl.restriction import PositiveRestricted, NegativeRestricted
from ufl.differentiation import SpatialDerivative, VariableDerivative


#--- Helper functions for product handling ---

def _mult(a, b):
    # Discover repeated indices, which results in index sums
    ai = a.free_indices()
    bi = b.free_indices()
    ii = ai + bi
    ri = repeated_indices(ii)

    # Pick out valid non-scalar products here (dot products):
    # - matrix-matrix (A*B, M*grad(u)) => A . B
    # - matrix-vector (A*v) => A . v
    s1, s2 = a.shape(), b.shape()
    r1, r2 = len(s1), len(s2)
    if r1 == 2 and r2 in (1, 2):
        ufl_assert(not ri, "Not expecting repeated indices in non-scalar product.")

        # Check for zero, simplifying early if possible
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = s1[:-1] + s2[1:]
            fi = single_indices(ii)
            idims = mergedicts((a.index_dimensions(), b.index_dimensions()))
            idims = subdict(idims, fi)
            return Zero(shape, fi, idims)

        # Return dot product in index notation
        ai = indices(a.rank()-1)
        bi = indices(b.rank()-1)
        k = indices(1)
        # Create an IndexSum over a Product
        s = a[ai+k]*b[k+bi]
        return as_tensor(s, ai+bi)

    elif not (r1 == 0 and r2 == 0):
        # Scalar - tensor product
        if r2 == 0:
            a, b = b, a
            s1, s2 = s2, s1

        # Check for zero, simplifying early if possible
        if isinstance(a, Zero) or isinstance(b, Zero):
            shape = s2
            fi = single_indices(ii)
            idims = mergedicts((a.index_dimensions(), b.index_dimensions()))
            idims = subdict(idims, fi)
            return Zero(shape, fi, idims)

        # Repeated indices are allowed, like in:
        #v[i]*M[i,:]

        # Apply product to scalar components
        ii = indices(b.rank())
        p = Product(a, b[ii])

        # Wrap as tensor again
        p = as_tensor(p, ii)

        # TODO: Should we apply IndexSum or as_tensor first?

        # Apply index sums
        for i in ri:
            p = IndexSum(p, i)

        return p

    # Scalar products use Product and IndexSum for implicit sums:
    p = Product(a, b)
    for i in ri:
        p = IndexSum(p, i)
    return p

#--- Extend Expr with algebraic operators ---

_valid_types = (Expr,) + python_scalar_types

def _mul(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    o = as_ufl(o)
    return _mult(self, o)
Expr.__mul__ = _mul

def _rmul(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    o = as_ufl(o)
    return _mult(o, self)
Expr.__rmul__ = _rmul

def _add(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(self, o)
Expr.__add__ = _add

def _radd(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(o, self)
Expr.__radd__ = _radd

def _sub(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(self, -o)
Expr.__sub__ = _sub

def _rsub(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Sum(o, -self)
Expr.__rsub__ = _rsub

def _div(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    sh = self.shape()
    if sh:
        ii = indices(len(sh))
        d = Division(self[ii], o)
        return as_tensor(d, ii)
    return Division(self, o)
Expr.__div__ = _div
Expr.__truediv__ = _div

def _rdiv(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Division(o, self)
Expr.__rdiv__ = _rdiv

def _pow(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Power(self, o)
Expr.__pow__ = _pow

def _rpow(self, o):
    if not isinstance(o, _valid_types):
        return NotImplemented
    return Power(o, self)
Expr.__rpow__ = _rpow

# TODO: Add Negated class for this? Might simplify reductions in Add.
def _neg(self):
    return -1*self
Expr.__neg__ = _neg

def _abs(self):
    return Abs(self)
Expr.__abs__ = _abs

#--- Extend Expr with restiction operators a("+"), a("-") ---

def _restrict(self, side):
    if side == "+":
        return PositiveRestricted(self)
    if side == "-":
        return NegativeRestricted(self)
    error("Invalid side %r in restriction operator." % side)
#Expr.__call__ = _restrict

def _call(self, arg, mapping=None):
    # Taking the restriction?
    if arg in ("+", "-"):
        ufl_assert(mapping is None, "Not expecting a mapping when taking restriction.")
        return _restrict(self, arg)

    # Evaluate expression at this particular coordinate,
    # with provided values for other terminals in mapping
    if mapping is None:
        mapping = {}
    component = ()
    index_values = StackDict()
    from ufl.algorithms import expand_derivatives
    if arg is None:
        dim = None
    elif isinstance(arg, (tuple, list)):
        dim = len(arg)
    else: # No type checking here...
        dim = 1
    f = expand_derivatives(self, dim)
    return f.evaluate(arg, mapping, component, index_values)
Expr.__call__ = _call

#--- Extend Expr with the transpose operation A.T ---

def _transpose(self):
    """Transposed a rank two tensor expression. For more general transpose
    operations of higher order tensor expressions, use indexing and Tensor."""
    return Transposed(self)
Expr.T = property(_transpose)

#--- Extend Expr with indexing operator a[i] ---

def analyse_key(ii, rank):
    """Takes something the user might input as an index tuple
    inside [], which could include complete slices (:) and
    ellipsis (...), and returns tuples of actual UFL index objects.

    The return value is a tuple (indices, axis_indices),
    each being a tuple of IndexBase instances.

    The return value 'indices' corresponds to all
    input objects of these types:
    - Index
    - FixedIndex
    - int => Wrapped in FixedIndex

    The return value 'axis_indices' corresponds to all
    input objects of these types:
    - Complete slice (:) => Replaced by a single new index
    - Ellipsis (...) => Replaced by multiple new indices
    """
    if not isinstance(ii, tuple):
        ii = (ii,)

    # Convert all indices to Index or FixedIndex objects.
    # If there is an ellipsis, split the indices into before and after.
    axis_indices = set()
    pre  = []
    post = []
    indexlist = pre
    for i in ii:
        if i == Ellipsis:
            # Switch from pre to post list when an ellipsis is encountered
            ufl_assert(indexlist is pre, "Found duplicate ellipsis.")
            indexlist = post
        else:
            # Convert index to a proper type
            if isinstance(i, int):
                idx = FixedIndex(i)
            elif isinstance(i, IndexBase):
                idx = i
            elif isinstance(i, slice):
                if i == slice(None):
                    idx = Index()
                    axis_indices.add(idx)
                else:
                    # TODO: Use ListTensor to support partial slices?
                    error("Partial slices not implemented, only complete slices like [:]")
            else:
                error("Can't convert this object to index: %r" % i)

            # Store index in pre or post list
            indexlist.append(idx)

    # Handle ellipsis as a number of complete slices,
    # that is create a number of new axis indices
    num_axis = rank - len(pre) - len(post)
    ellipsis_indices = indices(num_axis)
    axis_indices.update(ellipsis_indices)

    # Construct final tuples to return
    all_indices = tuple(chain(pre, ellipsis_indices, post))
    axis_indices = tuple(i for i in all_indices if i in axis_indices)
    return all_indices, axis_indices

def _getitem(self, key):
    # Analyse key, getting rid of slices and the ellipsis
    r = self.rank()
    indices, axis_indices = analyse_key(key, r)

    # Special case for foo[...] => foo
    if len(indices) == len(axis_indices):
        return self

    # Index self, yielding scalar valued expressions
    a = Indexed(self, indices)

    # Make a tensor from components designated by axis indices
    if axis_indices:
        a = as_tensor(a, axis_indices)

    # TODO: Should we apply IndexSum or as_tensor first?

    # Apply sum for each repeated index
    ri = repeated_indices(self.free_indices() + indices)
    for i in ri:
        a = IndexSum(a, i)

    # Check for zero (last so we can get indices etc from a)
    if isinstance(self, Zero):
        shape = a.shape()
        fi = a.free_indices()
        idims = subdict(a.index_dimensions(), fi)
        a = Zero(shape, fi, idims)

    return a
Expr.__getitem__ = _getitem

#--- Extend Expr with spatial differentiation operator a.dx(i) ---

def _dx(self, *ii):
    "Return the partial derivative with respect to spatial variable number i."

    d = self
    # Apply all derivatives
    for i in ii:
        d = SpatialDerivative(d, i)

    # Apply all implicit sums
    ri = repeated_indices(self.free_indices() + ii)
    for i in ri:
        d = IndexSum(d, i)

    return d
Expr.dx = _dx

#def _d(self, v):
#    "Return the partial derivative with respect to variable v."
#    # TODO: Maybe v can be an Indexed of a Variable, in which case we can use indexing to extract the right component?
#    return VariableDerivative(self, v)
#Expr.d = _d

