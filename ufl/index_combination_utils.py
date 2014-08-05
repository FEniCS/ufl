
from six.moves import zip
from six.moves import xrange as range
from itertools import chain

from collections import namedtuple

from ufl.assertions import ufl_assert
from ufl.log import error
from ufl.indexing import FixedIndex, Index, indices

free_tuple_type = namedtuple("free_tuple_type", "id pos dim")
fixed_tuple_type = namedtuple("fixed_tuple_type", "value pos dim")
slice_tuple_type = namedtuple("slice_tuple_type", "pos dim")


def unique_sorted_indices(indices):
    """Given a list of (id, dim) tuples already sorted by id,
    return a unique list with duplicates removed.
    Also checks that the dimensions of duplicates are matching.
    """
    newindices = []
    prev = (None, None)
    for i in indices:
        if i[0] != prev[0]:
            newindices.append(i)
            prev = i
        else:
            ufl_assert(i[1] == prev[1], "Nonmatching dimensions for free indices with same id!")
    return tuple(newindices)


def merge_unique_indices(afi, afid, bfi, bfid):
    """Merge two pairs of (index ids, index dimensions) sequences into one pair without duplicates.

    The id tuples afi, bfi are assumed already sorted by id.
    Given a list of (id, dim) tuples already sorted by id,
    return a unique list with duplicates removed.
    Also checks that the dimensions of duplicates are matching.
    """

    # TODO: Could be faster to loop over (afi,afid) and (bfi,bfid) in
    # parallel without presorting because they're both already sorted
    fiid = sorted(chain(zip(afi, afid), zip(bfi, bfid)))

    newfiid = []
    prev = (None, None)
    for curr in fiid:
        if curr[0] != prev[0]:
            newfiid.append(curr)
            prev = curr
        else:
            ufl_assert(curr[1] == prev[1], "Nonmatching dimensions for free indices with same id!")

    # Unpack into two tuples
    fii, fid = zip(*newfiid) if fiid else ((), ())
    return fii, fid


def remove_indices(fi, fid, rfi):
    """
    """
    if not rfi:
        return fi, fid

    rfip = sorted((r,p) for p, r in enumerate(rfi))

    nrfi = len(rfi)
    nfi = len(fi)
    shape = [None]*nrfi
    k = 0
    pos = 0
    newfiid = []
    while pos < nfi:
        rk = rfip[k][0]

        # Keep
        while fi[pos] < rk:
            newfiid.append((fi[pos], fid[pos]))
            pos += 1

        # Skip
        removed = 0
        while pos < nfi and fi[pos] == rk:
            shape[rfip[k][1]] = fid[pos]
            pos += 1
            removed += 1

        # Expecting to find each index from rfi in fi
        if not removed:
            error("Index to be removed ({0}) not part of indices ({1}).".format(rk, fi))

        # Next to remove
        k += 1
        if k == nrfi:
            # No more to remove, keep the rest
            if pos < nfi:
                newfiid.extend(zip(fi[pos:], fid[pos:]))
            break

    assert None not in shape

    # Unpack into two tuples
    fi, fid = zip(*newfiid) if newfiid else ((), ())

    return fi, fid, tuple(shape)


def create_slice_indices(component, shape):
    all_indices = []
    slice_indices = []

    for ind in component:
        if isinstance(ind, Index):
            all_indices.append(ind)
        elif isinstance(ind, FixedIndex):
            ufl_assert(int(ind) < shape[len(all_indices)],
                       "Index out of bounds.")
            all_indices.append(ind)
        elif isinstance(ind, int):
            ufl_assert(int(ind) < shape[len(all_indices)],
                       "Index out of bounds.")
            all_indices.append(FixedIndex(ind))
        elif isinstance(ind, slice):
            ufl_assert(ind == slice(None), "Only full slices (:) allowed.")
            i = Index()
            slice_indices.append(i)
            all_indices.append(i)
        elif ind == Ellipsis:
            er = len(shape) - len(component) + 1
            ii = indices(er)
            slice_indices.extend(ii)
            all_indices.extend(ii)
        else:
            error("Not expecting {0}.".format(ind))
    ufl_assert(len(all_indices) == len(shape),
               "Component and shape length don't match.")
    return tuple(all_indices), tuple(slice_indices)


def find_repeated_free_indices(free_indices):
    "Assumes free_indices is sorted."
    repeated = []
    for pos in range(1, len(free_indices)):
        if free_indices[pos-1].count() == free_indices[pos].count():
            repeated.append(free_indices[pos])
    return tuple(repeated)


def analyze_getitem_component(component, shape):
    free = []
    fixed = []
    for pos, ind in enumerate(component):
        if isinstance(ind, Index):
            free.append(free_tuple_type(ind.count(), pos, shape[pos]))
        elif isinstance(ind, (int, FixedIndex)):
            fixed.append(fixed_tuple_type(int(ind), pos, shape[pos]))
        else:
            error("Not expecting {0}.".format(ind))
    ufl_assert(len(free) + len(fixed) == len(shape),
               "Index and shape lengths don't match.")
    return tuple(sorted(free)), tuple(fixed)


def __find_repeated_free_indices(free_indices):
    "Assumes free_indices is sorted."
    r = len(free_indices)
    pos = 0
    ind = free_indices[pos]
    unrepeated = [(ind, pos)]
    repeated = []
    for pos in range(1,r):
        ind = free_indices[pos]
        if ind == unrepeated[-1][0]:
            ind, pos0 = unrepeated.pop()
            repeated.append((ind, pos0, pos))
        else:
            unrepeated.append((ind, pos))
    return tuple(unrepeated), tuple(repeated)


def find_repeated_free_indices2(free_indices1, free_indices2):
    repeated = []
    for pos1, ind1 in enumerate(free_indices1):
        for pos2, ind2 in enumerate(free_indices2):
            if ind2 > ind1:
                break
            elif ind2 == ind1:
                repeated.append((ind1, pos1, pos2))
    return tuple(repeated)


def remove_index(free_indices, free_index_dimensions, index):
    fi = []
    fid = []
    for pos, ind in enumerate(free_indices):
        if ind != index:
            fi.append(ind)
            fid.append(free_index_dimensions[pos])
    return tuple(fi), tuple(fid)


def extract_indices_shape(free_indices, free_index_dimensions, indices):
    fi = []
    fid = []
    shape = [None]*len(indices)
    for pos, ind in enumerate(free_indices):
        if ind in indices:
            shape[indices.index(ind)] = free_index_dimensions[pos]
        else:
            fi.append(ind)
            fid.append(free_index_dimensions[pos])
    return tuple(fi), tuple(fid), tuple(shape)


def slice_tuple(tup, slc):
    return tuple(tup[i] for i in slc)


def test():
    i, j = indices(2)

    shape = (2, 3, 4, 6, 7, 2, 8)
    component = (j, 2, i, Ellipsis, j, slice(None))
    print shape
    print component
    print

    # When indexing a tensor A[component], we need to figure out which indices are free or fixed or slices.

    # What's missing here is handling of A[subcomponent0, ..., subcomponent1] and A[subcomponent0, :, subcomponent1].
    # The way to do that is to just do as_tensor(A[subcomponent0, autocomponent0, subcomponent1], autocomponent0) to recover shape.
    all_indices, slice_indices = create_slice_indices(component, shape)
    print all_indices, slice_indices

    free, fixed, slices = analyze_getitem_component(component, shape)
    free_indices, free_index_positions, free_index_dimensions = zip(*free)
    print "free, fixed, slices"
    print free
    print fixed
    print slices
    print "free_indices, free_index_positions, free_index_dimensions"
    print free_indices
    print free_index_positions
    print free_index_dimensions
    print

    # When indexing a tensor A[component] and component has repeated indices, this implies summation.
    # This is analysed here:
    unrepeated, repeated = find_repeated_free_indices(free_indices)
    unrepeated_indices, unrepeated_index_positions = zip(*unrepeated)
    unrepeated_free_index_dimensions = slice_tuple(free_index_positions, unrepeated_index_positions)
    repeated_free_index_dimensions = slice_tuple(free_index_positions, unrepeated_index_positions)
    print unrepeated, repeated
    print unrepeated_indices, unrepeated_index_positions
    print unrepeated_free_index_dimensions, repeated_free_index_dimensions
    print

    # When multiplying two expressions with free indices, repeated indices across two tuples implies summation.
    # This is analysed here:
    repeated2 = find_repeated_free_indices2(free_indices, unrepeated_indices)
    print free_indices, unrepeated_indices, repeated2
    print

    # When an IndexSum wraps an expression, its indices are the indices of the expression with the summation index removed.
    free_indices = (1, 4, 4, 6)
    free_index_dimensions = (3, 2, 2, 5)
    index = 4
    free_indices2, free_index_dimensions2 = remove_index(free_indices, free_index_dimensions, index)
    print 'removed:'
    print free_indices, free_index_dimensions
    print free_indices2, free_index_dimensions2
    print

    # When as_tensor is applied, given indices are removed and their dimensions placed in a certain order to form the new shape.
    free_indices = (0,1,2,3)
    free_index_dimensions = (2,3,4,5)
    component_indices = (3,1,2)
    free_indices2, free_index_dimensions2, shape = extract_indices_shape(free_indices, free_index_dimensions, component_indices)
    print "as_tensor:"
    print free_indices, free_index_dimensions, component_indices
    print free_indices2, free_index_dimensions2, shape



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
def merge_nonoverlapping_indices(a, b):
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
    s = sorted(zip(ai + bi, aid + bid))
    if s:
        free_indices, index_dimensions = zip(*s)
        # Consistency checks
        ufl_assert(len(set(free_indices)) == len(free_indices), "Not expecting repeated indices.")
    else:
        free_indices, index_dimensions = (), ()
    return free_indices, index_dimensions


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
