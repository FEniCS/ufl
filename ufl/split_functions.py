"""Algorithm for splitting a Coefficient or Argument into subfunctions."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Anders Logg, 2008

from ufl.functionspace import FunctionSpace
from ufl.indexed import Indexed
from ufl.permutation import compute_indices
from ufl.tensors import ListTensor, as_matrix, as_vector
from ufl.utils.indexflattening import flatten_multiindex, shape_to_strides
from ufl.utils.sequences import product


def split(v):
    """Split a coefficient or argument.

    If v is a Coefficient or Argument in a mixed space, returns a tuple
    with the function components corresponding to the subelements.
    """
    domain = v.ufl_domain()

    # Default range is all of v
    begin = 0
    end = None

    if isinstance(v, Indexed):
        # Special case: split previous output of split again
        # Consistent with simple element, just return function in a tuple
        return (v,)

    elif isinstance(v, ListTensor):
        # Special case: split previous output of split again
        ops = v.ufl_operands
        if all(isinstance(comp, Indexed) for comp in ops):
            args = [comp.ufl_operands[0] for comp in ops]
            if all(args[0] == args[i] for i in range(1, len(args))):
                # Get innermost terminal here and its element
                v = args[0]
                # Get relevant range of v components
                (begin,) = ops[0].ufl_operands[1]
                (end,) = ops[-1].ufl_operands[1]
                begin = int(begin)
                end = int(end) + 1
            else:
                raise ValueError(f"Don't know how to split {v}.")
        else:
            raise ValueError(f"Don't know how to split {v}.")

    # Special case: simple element, just return function in a tuple
    element = v.ufl_element()
    if element.num_sub_elements == 0:
        assert end is None
        return (v,)

    if len(v.ufl_shape) != 1:
        raise ValueError(
            "Don't know how to split tensor valued mixed functions without flattened index space."
        )

    # Compute value size and set default range end
    value_size = v.ufl_function_space().value_size
    if end is None:
        end = value_size
    else:
        # Recursively dive into mixedelement in to subelement
        # corresponding to beginning of range
        j = begin
        while True:
            for e in element.sub_elements:
                if j < FunctionSpace(domain, e).value_size:
                    element = e
                    break
                j -= FunctionSpace(domain, e).value_size
            # Then break when we find the subelement that covers the whole range
            if FunctionSpace(domain, element).value_size == (end - begin):
                break

    # Build expressions representing the subfunction of v for each subelement
    offset = begin
    sub_functions = []
    for i, e in enumerate(element.sub_elements):
        # Get shape, size, indices, and v components
        # corresponding to subelement value
        shape = FunctionSpace(domain, e).value_shape
        strides = shape_to_strides(shape)
        rank = len(shape)
        sub_size = product(shape)
        subindices = [flatten_multiindex(c, strides) for c in compute_indices(shape)]
        components = [v[k + offset] for k in subindices]

        # Shape components into same shape as subelement
        if rank == 0:
            (subv,) = components
        elif rank <= 1:
            subv = as_vector(components)
        elif rank == 2:
            subv = as_matrix(
                [components[i * shape[1] : (i + 1) * shape[1]] for i in range(shape[0])]
            )
        else:
            raise ValueError(
                f"Don't know how to split functions with sub functions of rank {rank}."
            )

        offset += sub_size
        sub_functions.append(subv)

    if end != offset:
        raise ValueError(
            "Function splitting failed to extract components for whole intended range."
        )

    return tuple(sub_functions)
