

from six.moves import zip
from six.moves import xrange as range

from ufl.assertions import ufl_assert


def merge_indices(a, b):
    """Merge non-overlapping free indices into one representation."""

    ai = a.free_indices()
    bi = b.free_indices()
    aid = a.index_dimensions()
    bid = b.index_dimensions()

    free_indices = ai + bi
    index_dimensions = dict(aid)
    index_dimensions.update(bid)

    ufl_assert(len(index_dimensions) == len(free_indices), "Not expecting repeated indices.")

    return free_indices, index_dimensions


# Outer etc.
def new_merge_nonoverlapping_indices(a, b):
    """Merge non-overlapping free indices into one representation.

    Example:
      C[i,j,r,s] = outer(A[i,s], B[j,r])
      A, B -> (i,j,r,s), (idim,jdim,rdim,sdim)
    """

    # Extract input properties
    ai = a.ufl_free_indices
    bi = b.ufl_free_indices
    aid = a.ufl_index_dimensions
    bid = b.ufl_index_dimensions

    # Merge lists to return
    free_indices, index_dimensions = zip(*sorted(zip(ai + bi, aid + bid)))

    # Consistency checks
    ufl_assert(len(set(free_indices)) == len(free_indices), "Not expecting repeated indices.")

    return tuple(free_indices), tuple(index_dimensions)


# Product
def new_merge_overlapping_indices(a, b):
    """Merge overlapping free indices into one free and one repeated representation.

    Example:
      C[j,r] := A[i,j,k] * B[i,r,k]
      A, B -> (j,r), (jdim,rdim), (i,k), (idim,kdim)
    """

    # Extract input properties
    ai = a.ufl_free_indices
    bi = b.ufl_free_indices
    aid = a.ufl_index_dimensions
    bid = b.ufl_index_dimensions
    an = len(ai)
    bn = len(bi)

    # Lists to return
    free_indices = []
    index_dimensions = []
    repeated_indices = []
    repeated_index_dimensions = []

    # Find repeated indices, brute force version
    for i0 in range(an):
        for i1 in range(bn):
            if ai[i0] == bi[i1]:
                repeated_indices.append(ai[i0])
                repeated_index_dimensions.append(aid[i0])
                break

    # Collect only non-repeated indices, brute force version
    for i, d in sorted(zip(ai + bi, aid + bid)):
        if i not in repeated_indices:
            free_indices.append(i)
            index_dimensions.append(d)

    # Consistency checks
    ufl_assert(len(set(free_indices)) == len(free_indices), "Not expecting repeated indices left.")
    ufl_assert(len(free_indices) + 2*len(repeated_indices) == an + bn, "Expecting only twice repeated indices.")

    return tuple(free_indices), tuple(index_dimensions), tuple(repeated_indices), tuple(repeated_index_dimensions)


# MultiIndex
def new_extract_free_indices(ii, idims):
    """Given an unsorted Index list and an Index -> dim mapping, return free_indices and index_dimension tuples.

    Note: free_indices here may contain repeated indices! TODO: split here or somewhere else?

    Example:
        B[i,j] = A[0,j,1,i]
        A -> (i,j), (idim,jdim)
    """
    free = []
    for i in ii:
        if isinstance(i, Index):
            j = i.count()
            d = idims[i]
            free.append((j,d))

    # Sort and unzip lists to return
    free_indices, index_dimensions = zip(*sorted(free))

    return tuple(free_indices), tuple(index_dimensions)


# Indexed
def new_extract_repeated_indices(A, ii):
    """Given an unsorted Index list and an Index -> dim mapping, return free_indices and index_dimension tuples.

    Note: free_indices here may contain repeated indices! TODO: split here or somewhere else?

    Example:
        B[i,j] = A[0,j,k,1,i,k]
        A -> (i,j), (idim,jdim), (k,), (kdim,)
    """


# ComponentTensor
def new_slice_index_dimensions(a, i):
    """

    Example:
       B[i,k] = as_tensor(A[i,j,k], (i,k)) = as_tensor(a, ii)
       A, ii -> (i,k), (idim,kdim)
    """
    pass


# IndexSum
def new_foo_indices(a, i):
    """

    Example:
      B[j,k] = sum_j A[i,j,i,k]
      A, i -> (j,k), (jdim,kdim)
    """
    pass


# VariableDerivative
def new_foo_indices(a, i):
    """

    Example:
    """
    pass
