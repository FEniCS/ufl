"Algorithm for splitting a Coefficient or Argument into subfunctions."

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
# Modified by Anders Logg, 2008
#
# First added:  2008-03-14
# Last changed: 2011-06-22

from ufl.log import error
from ufl.assertions import ufl_assert
from ufl.common import product, EmptyDict
from ufl.finiteelement import MixedElement, TensorElement
from ufl.tensors import as_vector, as_matrix, as_tensor


def split(v):
    """UFL operator: If v is a Coefficient or Argument in a mixed space, returns
    a tuple with the function components corresponding to the subelements."""
    # Special case: simple element, just return function in a tuple
    element = v.element()
    if not isinstance(element, MixedElement):
        return (v,)

    if isinstance(element, TensorElement):
        s = element.symmetry()
        if s:
            # FIXME: How should this be defined? Should we return one subfunction
            # for each value component or only for those not mapped to another?
            # I think split should ignore the symmetry.
            error("Split not implemented for symmetric tensor elements.")

    # Compute value size
    value_size = product(element.value_shape())
    actual_value_size = value_size

    # Extract sub coefficient
    offset = 0
    sub_functions = []
    for i, e in enumerate(element.sub_elements()):
        shape = e.value_shape()
        rank = len(shape)

        if rank == 0:
            # This subelement is a scalar, always maps to a single value
            subv = v[offset]
            offset += 1

        elif rank == 1:
            # This subelement is a vector, always maps to a sequence of values
            sub_size, = shape
            components = [v[j] for j in range(offset, offset + sub_size)]
            subv = as_vector(components)
            offset += sub_size

        elif rank == 2:
            # This subelement is a tensor, possibly with symmetries, slightly more complicated...

            # Size of this subvalue
            sub_size = product(shape)

            # If this subelement is a symmetric element, subtract symmetric components
            s = None
            if isinstance(e, TensorElement):
                s = e.symmetry()
            s = s or EmptyDict
            # If we do this, we must fix the size computation in MixedElement.__init__ as well
            #actual_value_size -= len(s)
            #sub_size -= len(s)
            #print s
            # Build list of lists of value components
            components = []
            for ii in range(shape[0]):
                row = []
                for jj in range(shape[1]):
                    # Map component (i,j) through symmetry mapping
                    c = (ii, jj)
                    c = s.get(c, c)
                    i, j = c
                    # Extract component c of this subvalue from global tensor v
                    if v.rank() == 1:
                        # Mapping into a flattened vector
                        k = offset + i*shape[1] + j
                        component = v[k]
                        #print "k, offset, i, j, shape, component", k, offset, i, j, shape, component
                    elif v.rank() == 2:
                        # Mapping into a concatenated tensor (is this a figment of my imagination?)
                        error("Not implemented.")
                        row_offset, col_offset = 0, 0 # TODO
                        k = (row_offset + i, col_offset + j)
                        component = v[k]
                    row.append(component)
                components.append(row)

            # Make a matrix of the components
            subv = as_matrix(components)
            offset += sub_size

        else:
            # TODO: Handle rank > 2? Or is there such a thing?
            error("Don't know how to split functions with sub functions of rank %d (yet)." % rank)
            #for indices in compute_indices(shape):
            #    #k = offset + sum(i*s for (i,s) in izip(indices, shape[1:] + (1,)))
            #    vs.append(v[indices])

        sub_functions.append(subv)

    ufl_assert(actual_value_size == offset, "Logic breach in function splitting.")

    return tuple(sub_functions)
