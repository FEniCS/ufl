"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from itertools import accumulate, chain, repeat

import numpy

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.classes import (Jacobian, JacobianDeterminant, JacobianInverse,
                         ReferenceValue)
from ufl.core.multiindex import indices
from ufl.corealg.multifunction import MultiFunction, memoized_handler
from ufl.domain import extract_unique_domain
from ufl.tensors import as_tensor, as_vector
from ufl.utils.sequences import product


def sub_elements_with_mappings(element):
    "Return an ordered list of the largest subelements that have a defined mapping."
    if element.mapping() != "undefined":
        return [element]
    elements = []
    for subelm in element.sub_elements():
        if subelm.mapping() != "undefined":
            elements.append(subelm)
        else:
            elements.extend(sub_elements_with_mappings(subelm))
    return elements


def create_nested_lists(shape):
    if len(shape) == 0:
        return [None]
    elif len(shape) == 1:
        return [None] * shape[0]
    else:
        return [create_nested_lists(shape[1:]) for i in range(shape[0])]


def reshape_to_nested_list(components, shape):
    if len(shape) == 0:
        assert len(components) == 1
        return [components[0]]
    elif len(shape) == 1:
        assert len(components) == shape[0]
        return components
    else:
        n = product(shape[1:])
        return [reshape_to_nested_list(components[n * i:n * (i + 1)], shape[1:]) for i in range(shape[0])]


def apply_known_single_pullback(r, element):
    """Apply pullback with given mapping.

    :arg r: Expression wrapped in ReferenceValue
    :arg element: The element defining the mapping
    """
    # Need to pass in r rather than the physical space thing, because
    # the latter may be a ListTensor or similar, rather than a
    # Coefficient/Argument (in the case of mixed elements, see below
    # in apply_single_function_pullbacks), to which we cannot apply ReferenceValue
    mapping = element.mapping()
    domain = extract_unique_domain(r)
    if mapping == "physical":
        return r
    elif mapping == "identity" or mapping == "custom":
        return r
    elif mapping == "contravariant Piola":
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        transform = (1.0 / detJ) * J
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(r.ufl_shape) + 1)
        kj = (*k, j)
        f = as_tensor(transform[i, j] * r[kj], (*k, i))
        return f
    elif mapping == "covariant Piola":
        K = JacobianInverse(domain)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j = indices(len(r.ufl_shape) + 1)
        kj = (*k, j)
        f = as_tensor(K[j, i] * r[kj], (*k, i))
        return f
    elif mapping == "L2 Piola":
        detJ = JacobianDeterminant(domain)
        return r / detJ
    elif mapping == "double contravariant Piola":
        J = Jacobian(domain)
        detJ = JacobianDeterminant(J)
        transform = (1.0 / detJ) * J
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j, m, n = indices(len(r.ufl_shape) + 2)
        kmn = (*k, m, n)
        f = as_tensor((1.0 / detJ)**2 * J[i, m] * r[kmn] * J[j, n], (*k, i, j))
        return f
    elif mapping == "double covariant Piola":
        K = JacobianInverse(domain)
        # Apply transform "row-wise" to TensorElement(PiolaMapped, ...)
        *k, i, j, m, n = indices(len(r.ufl_shape) + 2)
        kmn = (*k, m, n)
        f = as_tensor(K[m, i] * r[kmn] * K[n, j], (*k, i, j))
        return f
    else:
        raise ValueError(f"Unsupported mapping: {mapping}.")


def apply_single_function_pullbacks(r, element):
    """Apply an appropriate pullback to something in physical space

    :arg r: An expression wrapped in ReferenceValue.
    :arg element: The element this expression lives in.
    :returns: a pulled back expression."""
    mapping = element.mapping()
    if r.ufl_shape != element.reference_value_shape():
        raise ValueError(
            f"Expecting reference space expression with shape '{element.reference_value_shape()}', got '{r.ufl_shape}'")
    if mapping in {"physical", "identity",
                   "contravariant Piola", "covariant Piola",
                   "double contravariant Piola", "double covariant Piola",
                   "L2 Piola", "custom"}:
        # Base case in recursion through elements. If the element
        # advertises a mapping we know how to handle, do that
        # directly.
        f = apply_known_single_pullback(r, element)
        if f.ufl_shape != element.value_shape():
            raise ValueError(f"Expecting pulled back expression with shape '{element.value_shape()}', got '{f.ufl_shape}'")
        return f
    elif mapping in {"symmetries", "undefined"}:
        # Need to pull back each unique piece of the reference space thing
        gsh = element.value_shape()
        rsh = r.ufl_shape
        if mapping == "symmetries":
            subelem = element.sub_elements()[0]
            fcm = element.flattened_sub_element_mapping()
            offsets = (product(subelem.reference_value_shape()) * i for i in fcm)
            elements = repeat(subelem)
        else:
            elements = sub_elements_with_mappings(element)
            # Python >= 3.8 has an initial keyword argument to
            # accumulate, but 3.7 does not.
            offsets = chain([0],
                            accumulate(product(e.reference_value_shape())
                                       for e in elements))
        rflat = as_vector([r[idx] for idx in numpy.ndindex(rsh)])
        g_components = []
        # For each unique piece in reference space, apply the appropriate pullback
        for offset, subelem in zip(offsets, elements):
            sub_rsh = subelem.reference_value_shape()
            rm = product(sub_rsh)
            rsub = [rflat[offset + i] for i in range(rm)]
            rsub = as_tensor(numpy.asarray(rsub).reshape(sub_rsh))
            rmapped = apply_single_function_pullbacks(rsub, subelem)
            # Flatten into the pulled back expression for the whole thing
            g_components.extend([rmapped[idx]
                                 for idx in numpy.ndindex(rmapped.ufl_shape)])
        # And reshape appropriately
        f = as_tensor(numpy.asarray(g_components).reshape(gsh))
        if f.ufl_shape != element.value_shape():
            raise ValueError(f"Expecting pulled back expression with shape '{element.value_shape()}', got '{f.ufl_shape}'")
        return f
    else:
        raise ValueError(f"Unsupported mapping type: {mapping}")


class FunctionPullbackApplier(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    @memoized_handler
    def form_argument(self, o):
        # Represent 0-derivatives of form arguments on reference
        # element
        f = apply_single_function_pullbacks(ReferenceValue(o), o.ufl_element())
        assert f.ufl_shape == o.ufl_shape
        return f


def apply_function_pullbacks(expr):
    """Change representation of coefficients and arguments in expression
    by applying Piola mappings where applicable and representing all
    form arguments in reference value.

    @param expr:
        An Expr.
    """
    return map_integrand_dags(FunctionPullbackApplier(), expr)
