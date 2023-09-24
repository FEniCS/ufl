"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import ReferenceValue
from ufl.corealg.multifunction import MultiFunction, memoized_handler


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
    if r.ufl_shape != element.reference_value_shape:
        raise ValueError(
            f"Expecting reference space expression with shape '{element.reference_value_shape}', "
            f"got '{r.ufl_shape}'")
    f = apply_known_single_pullback(r, element)
    if f.ufl_shape != element.value_shape:
        raise ValueError(f"Expecting pulled back expression with shape '{element.value_shape}', "
                         f"got '{f.ufl_shape}'")
    return f


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
