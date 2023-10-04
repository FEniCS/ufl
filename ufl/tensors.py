"""Classes used to group scalar expressions into expressions with rank > 0."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Massimiliano Leoni, 2016.

from ufl.core.ufl_type import ufl_type
from ufl.core.expr import Expr
from ufl.core.operator import Operator
from ufl.constantvalue import as_ufl, Zero
from ufl.core.multiindex import Index, FixedIndex, MultiIndex, indices
from ufl.indexed import Indexed
from ufl.index_combination_utils import remove_indices


# --- Classes representing tensors of UFL expressions ---

@ufl_type(is_shaping=True, num_ops="varying", inherit_indices_from_operand=0)
class ListTensor(Operator):
    """Wraps a list of expressions into a tensor valued expression of one higher rank."""

    __slots__ = ()

    def __new__(cls, *expressions):
        """Create a new ListTensor."""
        # All lists and tuples should already be unwrapped in
        # as_tensor
        if any(not isinstance(e, Expr) for e in expressions):
            raise ValueError("Expecting only UFL expressions in ListTensor constructor.")

        # Get properties of the first expression
        e0 = expressions[0]
        sh = e0.ufl_shape
        fi = e0.ufl_free_indices
        fid = e0.ufl_index_dimensions

        # Obviously, each subexpression must have the same shape
        if any(sh != e.ufl_shape for e in expressions[1:]):
            raise ValueError("Cannot create a tensor by joining subexpressions with different shapes.")
        if any(fi != e.ufl_free_indices for e in expressions[1:]):
            raise ValueError("Cannot create a tensor where the components have different free indices.")
        if any(fid != e.ufl_index_dimensions for e in expressions[1:]):
            raise ValueError("Cannot create a tensor where the components have different free index dimensions.")

        # Simplify to Zero if possible
        if all(isinstance(e, Zero) for e in expressions):
            shape = (len(expressions),) + sh
            return Zero(shape, fi, fid)

        return Operator.__new__(cls)

    def __init__(self, *expressions):
        """Initialise."""
        Operator.__init__(self, expressions)

        # Checks
        indexset = set(self.ufl_operands[0].ufl_free_indices)
        if not all(not (indexset ^ set(e.ufl_free_indices)) for e in self.ufl_operands):
            raise ValueError("Can't combine subtensor expressions with different sets of free indices.")

    @property
    def ufl_shape(self):
        """Get the UFL shape."""
        return (len(self.ufl_operands),) + self.ufl_operands[0].ufl_shape

    def evaluate(self, x, mapping, component, index_values, derivatives=()):
        """Evaluate."""
        if len(component) != len(self.ufl_shape):
            raise ValueError(
                "Can only evaluate scalars, expecting a component "
                "tuple of length {len(self.ufl_shape)}, not {component}.")
        a = self.ufl_operands[component[0]]
        component = component[1:]
        if derivatives:
            return a.evaluate(x, mapping, component, index_values, derivatives)
        else:
            return a.evaluate(x, mapping, component, index_values)

    def __getitem__(self, key):
        """Get an item."""
        origkey = key

        if isinstance(key, MultiIndex):
            key = key.indices()
        if not isinstance(key, tuple):
            key = (key,)
        k = key[0]
        if isinstance(k, (int, FixedIndex)):
            sub = self.ufl_operands[int(k)]
            return sub if len(key) == 1 else sub[key[1:]]

        return Expr.__getitem__(self, origkey)

    def __str__(self):
        """Format as a string."""
        def substring(expressions, indent):
            ind = " " * indent
            if any(isinstance(e, ListTensor) for e in expressions):
                substrings = []
                for e in expressions:
                    if isinstance(e, ListTensor):
                        substrings.append(substring(e.ufl_operands, indent + 2))
                    else:
                        substrings.append(str(e))
                s = (",\n" + ind).join(substrings)
                return "%s[\n%s%s\n%s]" % (ind, ind, s, ind)
            else:
                s = ", ".join(str(e) for e in expressions)
                return "%s[%s]" % (ind, s)
        return substring(self.ufl_operands, 0)


@ufl_type(is_shaping=True, num_ops="varying")
class ComponentTensor(Operator):
    """Maps the free indices of a scalar valued expression to tensor axes."""

    __slots__ = ("ufl_shape", "ufl_free_indices", "ufl_index_dimensions")

    def __new__(cls, expression, indices):
        """Create a new ComponentTensor."""
        # Simplify
        if isinstance(expression, Zero):
            fi, fid, sh = remove_indices(expression.ufl_free_indices,
                                         expression.ufl_index_dimensions,
                                         [ind.count() for ind in indices])
            return Zero(sh, fi, fid)

        # Construct
        return Operator.__new__(cls)

    def __init__(self, expression, indices):
        """Initialise."""
        if not isinstance(expression, Expr):
            raise ValueError("Expecting ufl expression.")
        if expression.ufl_shape != ():
            raise ValueError("Expecting scalar valued expression.")
        if not isinstance(indices, MultiIndex):
            raise ValueError("Expecting a MultiIndex.")
        if not all(isinstance(i, Index) for i in indices):
            raise ValueError(f"Expecting sequence of Index objects, not {indices._ufl_err_str_()}.")

        Operator.__init__(self, (expression, indices))

        fi, fid, sh = remove_indices(expression.ufl_free_indices,
                                     expression.ufl_index_dimensions,
                                     [ind.count() for ind in indices])
        self.ufl_free_indices = fi
        self.ufl_index_dimensions = fid
        self.ufl_shape = sh

    def _ufl_expr_reconstruct_(self, expressions, indices):
        """Reconstruct."""
        # Special case for simplification as_tensor(A[ii], ii) -> A
        if isinstance(expressions, Indexed):
            A, ii = expressions.ufl_operands
            if indices == ii:
                return A
        return Operator._ufl_expr_reconstruct_(self, expressions, indices)

    def indices(self):
        """Get indices."""
        return self.ufl_operands[1]

    def evaluate(self, x, mapping, component, index_values):
        """Evaluate."""
        indices = self.ufl_operands[1]
        a = self.ufl_operands[0]

        if len(indices) != len(component):
            raise ValueError("Expecting a component matching the indices tuple.")

        # Map component to indices
        for i, c in zip(indices, component):
            index_values.push(i, c)

        a = a.evaluate(x, mapping, (), index_values)

        for _ in component:
            index_values.pop()

        return a

    def __str__(self):
        """Format as a string."""
        return "{ A | A_{%s} = %s }" % (self.ufl_operands[1], self.ufl_operands[0])


# --- User-level functions to wrap expressions in the correct way ---

def numpy2nestedlists(arr):
    """Convert Numpy array to a nested list."""
    from numpy import ndarray
    if not isinstance(arr, ndarray):
        return arr
    return [numpy2nestedlists(arr[k]) for k in range(arr.shape[0])]


def _as_list_tensor(expressions):
    """Convert to a list tensor."""
    if isinstance(expressions, (list, tuple)):
        expressions = [_as_list_tensor(e) for e in expressions]
        return ListTensor(*expressions)
    else:
        return as_ufl(expressions)


def from_numpy_to_lists(expressions):
    """Convert Numpy array to lists."""
    try:
        import numpy
        if isinstance(expressions, numpy.ndarray):
            if expressions.shape == ():
                # Unwrap scalar ndarray
                return expressions.item()
            else:
                expressions = numpy2nestedlists(expressions)
    except Exception:
        pass
    return expressions


def as_tensor(expressions, indices=None):
    """Make a tensor valued expression.

    This works in two different ways, by using indices or lists.

    1) Returns :math:`A` such that :math:`A` [*indices*] = *expressions*.
    If *indices* are provided, *expressions* must be a scalar
    valued expression with all the provided indices among
    its free indices. This operator will then map each of these
    indices to a tensor axis, thereby making a tensor valued
    expression from a scalar valued expression with free indices.

    2) Returns :math:`A` such that :math:`A[k,...]` = *expressions*[k].
    If no indices are provided, *expressions* must be a list
    or tuple of expressions. The expressions can also consist
    of recursively nested lists to build higher rank tensors.
    """
    if indices is None:
        # Allow as_tensor(as_tensor(A)) and as_vector(as_vector(v)) in user code
        if isinstance(expressions, Expr):
            return expressions

        # Support numpy array, but avoid importing numpy if not needed
        if not isinstance(expressions, (list, tuple)):
            expressions = from_numpy_to_lists(expressions)

        # Sanity check
        if not isinstance(expressions, (list, tuple, Expr)):
            raise ValueError("Expecting nested list or tuple.")

        # Recursive conversion from nested lists to nested ListTensor
        # objects
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

        indices = MultiIndex(indices)

        # Special case for simplification as_tensor(A[ii], ii) -> A
        if isinstance(expressions, Indexed):
            A, ii = expressions.ufl_operands
            if indices.indices() == ii.indices():
                return A

        # Make a tensor from given scalar expression with free indices
        return ComponentTensor(expressions, indices)


def as_matrix(expressions, indices=None):
    """As *as_tensor()*, but limited to rank 2 tensors."""
    if indices is None:
        # Allow as_matrix(as_matrix(A)) in user code
        if isinstance(expressions, Expr):
            if len(expressions.ufl_shape) != 2:
                raise ValueError("Expecting rank 2 tensor.")
            return expressions

        # To avoid importing numpy unneeded, it's quite slow...
        if not isinstance(expressions, (list, tuple)):
            expressions = from_numpy_to_lists(expressions)

        # Check for expected list structure
        if not isinstance(expressions, (list, tuple)):
            raise ValueError("Expecting nested list or tuple of Exprs.")
        if not isinstance(expressions[0], (list, tuple)):
            raise ValueError("Expecting nested list or tuple of Exprs.")
    else:
        if len(indices) != 2:
            raise ValueError("Expecting exactly two indices.")

    return as_tensor(expressions, indices)


def as_vector(expressions, index=None):
    """As ``as_tensor()``, but limited to rank 1 tensors."""
    if index is None:
        # Allow as_vector(as_vector(v)) in user code
        if isinstance(expressions, Expr):
            if len(expressions.ufl_shape) != 1:
                raise ValueError("Expecting rank 1 tensor.")
            return expressions

        # To avoid importing numpy unneeded, it's quite slow...
        if not isinstance(expressions, (list, tuple)):
            expressions = from_numpy_to_lists(expressions)

        # Check for expected list structure
        if not isinstance(expressions, (list, tuple)):
            raise ValueError("Expecting nested list or tuple of Exprs.")
    else:
        if not isinstance(index, Index):
            raise ValueError("Expecting a single Index object.")
        index = (index,)

    return as_tensor(expressions, index)


def as_scalar(expression):
    """As scalar.

    Given a scalar or tensor valued expression A, returns either of the tuples::
      (a,b) = (A, ())
      (a,b) = (A[indices], indices)
    such that a is always a scalar valued expression.
    """
    ii = indices(len(expression.ufl_shape))
    if ii:
        expression = expression[ii]
    return expression, ii


def as_scalars(*expressions):
    """As scalars.

    Given multiple scalar or tensor valued expressions A, returns either of the tuples::
      (a,b) = (A, ())
      (a,b) = ([A[0][indices], ..., A[-1][indices]], indices)
    such that a is always a list of scalar valued expressions.
    """
    ii = indices(len(expressions[0].ufl_shape))
    if ii:
        expressions = [expression[ii] for expression in expressions]
    return expressions, ii


def unit_list(i, n):
    """Create a list of zeros where the ith entry is 1."""
    return [(1 if i == j else 0) for j in range(n)]


def unit_list2(i, j, n):
    """Creage a two dimensional list of zeros where the (i,j)th entry is 1."""
    return [[(1 if (i == i0 and j == j0) else 0) for j0 in range(n)] for i0 in range(n)]


def unit_vector(i, d):
    """A constant unit vector in direction *i* with dimension *d*."""
    return as_vector(unit_list(i, d))


def unit_vectors(d):
    """A tuple of constant unit vectors in all directions with dimension *d*."""
    return tuple(unit_vector(i, d) for i in range(d))


def unit_matrix(i, j, d):
    """A constant unit matrix in direction *i*,*j* with dimension *d*."""
    return as_matrix(unit_list2(i, j, d))


def unit_matrices(d):
    """A tuple of constant unit matrices in all directions with dimension *d*."""
    return tuple(unit_matrix(i, j, d) for i in range(d) for j in range(d))


def unit_indexed_tensor(shape, component):
    """Unit indexed tensor."""
    from ufl.constantvalue import Identity
    from ufl.operators import outer  # a bit of circular dependency issue here
    r = len(shape)
    if r == 0:
        return 0, ()
    jj = indices(r)
    es = []
    for i in range(r):
        s = shape[i]
        c = component[i]
        j = jj[i]
        e = Identity(s)[c, j]
        es.append(e)
    E = es[0]
    for e in es[1:]:
        E = outer(E, e)
    return E, jj


def unwrap_list_tensor(lt):
    """Unwrap a list tensor."""
    components = []
    sh = lt.ufl_shape
    subs = lt.ufl_operands
    if len(sh) == 1:
        for s in range(sh[0]):
            components.append(((s,), subs[s]))
    else:
        for s, sub in enumerate(subs):
            for c, v in unwrap_list_tensor(sub):
                components.append(((s,) + c, v))
    return components
