"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from itertools import repeat

import numpy

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import ReferenceValue
from ufl.corealg.multifunction import MultiFunction, memoized_handler
from ufl.pull_back import NonStandardPullBackException
from ufl.tensors import as_tensor, as_vector


def apply_known_single_pullback(r, element):
    """Apply pullback with given mapping.

    Args:
        r: Expression wrapped in ReferenceValue
        element: The element defining the mapping
    """
    # Need to pass in r rather than the physical space thing, because
    # the latter may be a ListTensor or similar, rather than a
    # Coefficient/Argument (in the case of mixed elements, see below
    # in apply_single_function_pullbacks), to which we cannot apply ReferenceValue
    return element.pull_back.apply(r)


def apply_single_function_pullbacks(r, element):
    """Apply an appropriate pullback to something in physical space.

    Args:
        r: An expression wrapped in ReferenceValue.
        element: The element this expression lives in.

    Returns:
        a pulled back expression.
    """
    mapping = element.pull_back
    if r.ufl_shape != element.reference_value_shape:
        raise ValueError(
            f"Expecting reference space expression with shape '{element.reference_value_shape}', "
            f"got '{r.ufl_shape}'")
    try:
        # Base case in recursion through elements. If the element
        # advertises a mapping we know how to handle, do that
        # directly.
        f = apply_known_single_pullback(r, element)
        if f.ufl_shape != element.value_shape:
            raise ValueError(f"Expecting pulled back expression with shape '{element.value_shape}', "
                             f"got '{f.ufl_shape}'")
        return f
    except NonStandardPullBackException:
        # TODO: Move this code to pull_back.py
        # Need to pull back each unique piece of the reference space thing
        gsh = element.value_shape
        rsh = r.ufl_shape
        # if mapping == "symmetries":
        subelem = element.sub_elements[0]
        fcm = element.flattened_sub_element_mapping()
        offsets = (subelem.reference_value_size * i for i in fcm)
        elements = repeat(subelem)
        rflat = as_vector([r[idx] for idx in numpy.ndindex(rsh)])
        g_components = []
        # For each unique piece in reference space, apply the appropriate pullback
        for offset, subelem in zip(offsets, elements):
            sub_rsh = subelem.reference_value_shape
            rm = subelem.reference_value_size
            rsub = [rflat[offset + i] for i in range(rm)]
            rsub = as_tensor(numpy.asarray(rsub).reshape(sub_rsh))
            rmapped = apply_single_function_pullbacks(rsub, subelem)
            # Flatten into the pulled back expression for the whole thing
            g_components.extend([rmapped[idx]
                                 for idx in numpy.ndindex(rmapped.ufl_shape)])
        # And reshape appropriately
        f = as_tensor(numpy.asarray(g_components).reshape(gsh))
        if f.ufl_shape != element.value_shape:
            raise ValueError(f"Expecting pulled back expression with shape '{element.value_shape}', "
                             f"got '{f.ufl_shape}'")
        return f
    else:
        raise ValueError(f"Unsupported mapping type: {mapping}")


class FunctionPullbackApplier(MultiFunction):
    """A pull back applier."""

    def __init__(self):
        """Initalise."""
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        """Apply to a terminal."""
        return t

    @memoized_handler
    def form_argument(self, o):
        """Apply to a form_argument."""
        # Represent 0-derivatives of form arguments on reference
        # element
        f = apply_single_function_pullbacks(ReferenceValue(o), o.ufl_element())
        assert f.ufl_shape == o.ufl_shape
        return f


def apply_function_pullbacks(expr):
    """Change representation of coefficients and arguments in an expression.

    Applies Piola mappings where applicable and represents all
    form arguments in reference value.

    Args:
        expr: An Expression
    """
    return map_integrand_dags(FunctionPullbackApplier(), expr)
