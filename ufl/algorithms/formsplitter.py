"""Extract part of a form in a mixed FunctionSpace."""

# Copyright (C) 2016-2024 Chris Richardson and Lawrence Mitchell
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Cecile Daversin-Catty, 2018
# Modified by Jørgen S. Dokken, 2025

import numpy as np

from ufl.algorithms.map_integrands import map_expr_dag, map_integrand_dags
from ufl.argument import Argument
from ufl.classes import FixedIndex, ListTensor
from ufl.constantvalue import Zero
from ufl.corealg.multifunction import MultiFunction
from ufl.form import Form
from ufl.functionspace import FunctionSpace
from ufl.tensors import as_vector


class FormSplitter(MultiFunction):
    """Form splitter."""

    def __init__(self, replace_argument: bool = True):
        """Initialize form splitter.

        Args:
            replace_argument: If True, replace the argument by a new argument
                in the sub-function space. If False, keep the original argument.
                This is useful for instance when diagonalizing a form with a mixed-element
                form, where we want to keep the original argument.
        """
        MultiFunction.__init__(self)
        self.idx = [None, None]
        self.replace_argument = replace_argument

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
            counter = 0
            for i, sub_elem in enumerate(sub_elements):
                Q_i = FunctionSpace(dom, sub_elem)
                a = Argument(Q_i, obj.number(), part=obj.part())
                if self.replace_argument:
                    if i == self.idx[obj.number()]:
                        args.extend(a[j] for j in np.ndindex(a.ufl_shape))
                    else:
                        args.extend(Zero() for _ in np.ndindex(a.ufl_shape))
                else:
                    # If we are not replacing the argument, we need to insert
                    # the original argument at the right place in the vector.
                    # Mixed elements are flattened, thus we need to keep track of
                    # the position in the flattened vector.
                    if i == self.idx[obj.number()]:
                        if a.ufl_shape == ():
                            args.append(obj[counter])
                        else:
                            args.extend(
                                obj[counter + j] for j, _ in enumerate(np.ndindex(a.ufl_shape))
                            )
                    else:
                        args.extend(Zero() for _ in np.ndindex(a.ufl_shape))
                    counter += int(np.prod(a.ufl_shape))
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
    form,
    i: int | None = None,
    j: int | None = None,
    arity: int | None = None,
    replace_argument: bool = True,
) -> Form | tuple[Form | None, ...] | tuple[tuple[Form | None, ...], ...]:
    """Extract blocks of a form.

    If arity is 0, returns the form.
    If arity is 1, return the ith block. If ``i`` is ``None``, return all blocks.
    If arity is 2, return the ``(i,j)`` entry. If ``j`` is ``None``, return the ith row.

    If neither `i` nor `j` are set, return all blocks (as a scalar, vector or tensor).

    Args:
        form:
            A form
        i:
            Index of the block to extract. If set to ``None``, ``j`` must be None.
        j:
            Index of the block to extract.
        arity:
            Arity of the form. If not set, it will be inferred from the form.
        replace_argument:
            If True, replace the argument by a new argument
            in the (collapsed) sub-function space. If False, keep the original argument.
    """
    if i is None and j is not None:
        raise RuntimeError(f"Cannot extract block with {j=} and {i=}.")

    fs = FormSplitter(replace_argument=replace_argument)
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
            # If form has no sub elements, return the form itself.
            if num_sub_elements == 0:
                return form
            forms = []
            for pi in range(num_sub_elements):
                form_i: list[object | None] = []
                for pj in range(num_sub_elements):
                    f = fs.split(form, pi, pj)
                    if f.empty():
                        form_i.append(None)
                    else:
                        form_i.append(f)
                forms.append(tuple(form_i))
            return tuple(forms)  # type: ignore[return-value]
        else:
            return fs.split(form, i, j)

    # If mixed function space, each argument has sub-elements
    num_parts = max(parts) + 1
    forms = [None] * num_parts  # type: ignore
    if arity == 2:
        for k in range(num_parts):
            forms[k] = [None] * num_parts  # type: ignore
    for pi in range(num_parts):
        if arity > 1:
            for pj in range(num_parts):
                f = fs.split(form, pi, pj)
                # Ignore empty forms and rank 0 or 1 forms
                if f.empty() or len(f.arguments()) != 2:
                    pass
                else:
                    forms[pi][pj] = f  # type: ignore
        else:
            f = fs.split(form, pi)
            # Ignore empty forms and bilinear forms
            if f.empty() or len(f.arguments()) != 1:
                pass
            else:
                forms[pi] = f

    if i is not None:
        if (num_rows := len(forms)) <= i:
            raise RuntimeError(f"Cannot extract block {i} from form with {num_rows} blocks.")
        if arity > 1 and j is not None:
            if (num_cols := len(forms[i])) <= j:
                raise RuntimeError(
                    f"Cannot extract block {i},{j} from form with {num_rows}x{num_cols} blocks."
                )
            return forms[i][j]  # type: ignore[return-value]
        else:
            return forms[i]  # type: ignore[return-value]
    else:
        if arity == 1:
            return tuple(forms)  # type: ignore[return-value]
        else:
            return tuple(tuple(row) for row in forms)  # type: ignore[return-value]
