"""Extract part of a form in a mixed FunctionSpace."""

# Copyright (C) 2016-2024 Chris Richardson and Lawrence Mitchell
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Cecile Daversin-Catty, 2018

from typing import Optional

import numpy as np

from ufl.algorithms.map_integrands import map_expr_dag, map_integrand_dags
from ufl.argument import Argument
from ufl.classes import FixedIndex, ListTensor
from ufl.constantvalue import Zero
from ufl.corealg.multifunction import MultiFunction
from ufl.functionspace import FunctionSpace
from ufl.tensors import as_vector


class FormSplitter(MultiFunction):
    """Form splitter."""

    def split(self, form, ix, iy=None):
        """Split form based on the argument part/number."""
        # Remember which block to extract
        self.idx = [ix, iy]
        return map_integrand_dags(self, form)

    def argument(self, obj):
        """Apply to argument."""
        if obj.part() is not None:
            # Mixed element built from MixedFunctionSpace,
            # whose sub-function spaces are indexed by obj.part()
            if self.idx[obj.number()] is None:
                return Zero(obj.ufl_shape)
            if obj.part() == self.idx[obj.number()]:
                return obj
            else:
                return Zero(obj.ufl_shape)
        else:
            # Mixed element built from MixedElement,
            # whose sub-elements need their function space to be created
            Q = obj.ufl_function_space()
            dom = Q.ufl_domain()
            sub_elements = obj.ufl_element().sub_elements

            # If not a mixed element, do nothing
            if len(sub_elements) == 0:
                return obj

            args = []
            for i, sub_elem in enumerate(sub_elements):
                Q_i = FunctionSpace(dom, sub_elem)
                a = Argument(Q_i, obj.number(), part=obj.part())

                if i == self.idx[obj.number()]:
                    args.extend(a[j] for j in np.ndindex(a.ufl_shape))
                else:
                    args.extend(Zero() for j in np.ndindex(a.ufl_shape))

            return as_vector(args)

    def indexed(self, o, child, multiindex):
        """Extract indexed entry if multindices are fixed.

        This avoids tensors like (v_0, 0)[1] to be created.
        """
        indices = multiindex.indices()
        if isinstance(child, ListTensor) and all(isinstance(i, FixedIndex) for i in indices):
            if len(indices) == 1:
                return child[indices[0]]
            elif len(indices) == len(child.ufl_operands) and all(
                k == int(i) for k, i in enumerate(indices)
            ):
                return child
            else:
                return ListTensor(*(child[i] for i in indices))
        return self.expr(o, child, multiindex)

    def multi_index(self, obj):
        """Apply to multi_index."""
        return obj

    def restricted(self, o):
        """Apply to a restricted function."""
        # If we hit a restriction first apply form splitter to argument, then check for zero
        op_split = map_expr_dag(self, o.ufl_operands[0])
        if isinstance(op_split, Zero):
            return op_split
        else:
            return op_split(o._side)

    expr = MultiFunction.reuse_if_untouched


def extract_blocks(
    form, i: Optional[int] = None, j: Optional[int] = None, arity: Optional[int] = None
):
    """Extract blocks of a form.

    If arity is 0, returns the form.
    If arity is 1, return the ith block. If ``i`` is ``None``, return all blocks.
    If arity is 2, return the ``(i,j)`` entry. If ``j`` is ``None``, return the ith row.

    If neither `i` nor `j` are set, return all blocks (as a scalar, vector or tensor).

    Args:
        form: A form
        i: Index of the block to extract. If set to ``None``, ``j`` must be None.
        j: Index of the block to extract.
        arity: Arity of the form. If not set, it will be inferred from the form.
    """
    if i is None and j is not None:
        raise RuntimeError(f"Cannot extract block with {j=} and {i=}.")

    fs = FormSplitter()
    arguments = form.arguments()

    if arity is None:
        numbers = tuple(sorted(set(a.number() for a in arguments)))
        arity = len(numbers)

    assert arity <= 2
    if arity == 0:
        return (form,)

    # If mixed element, each argument has no sub-elements
    parts = tuple(sorted(set(part for a in arguments if (part := a.part()) is not None)))
    if parts == ():
        if i is None and j is None:
            num_sub_elements = arguments[0].ufl_element().num_sub_elements
            forms = []
            for pi in range(num_sub_elements):
                form_i = []
                for pj in range(num_sub_elements):
                    f = fs.split(form, pi, pj)
                    if f.empty():
                        form_i.append(None)
                    else:
                        form_i.append(f)
                forms.append(tuple(form_i))
            return tuple(forms)
        else:
            return fs.split(form, i, j)

    # If mixed function space, each argument has sub-elements
    forms = []
    num_parts = len(parts)
    for pi in range(num_parts):
        form_i = []
        if arity > 1:
            for pj in range(num_parts):
                f = fs.split(form, pi, pj)
                # Ignore empty forms and rank 0 or 1 forms
                if f.empty() or len(f.arguments()) != 2:
                    form_i.append(None)
                else:
                    form_i.append(f)
            forms.append(tuple(form_i))
        else:
            f = fs.split(form, pi)
            # Ignore empty forms and bilinear forms
            if f.empty() or len(f.arguments()) != 1:
                forms.append(None)
            else:
                forms.append(f)

    try:
        forms_tuple = tuple(forms)
    except TypeError:
        # Only one form returned
        forms_tuple = (forms,)
    if i is not None:
        if (num_rows := len(forms_tuple)) <= i:
            raise RuntimeError(f"Cannot extract block {i} from form with {num_rows} blocks.")
        if arity > 1 and j is not None:
            if (num_cols := len(forms_tuple[i])) <= j:
                raise RuntimeError(
                    f"Cannot extract block {i},{j} from form with {num_rows}x{num_cols} blocks."
                )
            return forms_tuple[i][j]
        else:
            return forms_tuple[i]
    else:
        return forms_tuple
