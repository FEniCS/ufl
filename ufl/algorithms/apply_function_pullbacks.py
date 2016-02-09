# -*- coding: utf-8 -*-
"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2008-2015 Martin Sandve Alnæs
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

from six.moves import xrange as range

from ufl.log import error
from ufl.assertions import ufl_assert

from ufl.core.multiindex import indices
from ufl.corealg.multifunction import MultiFunction, memoized_handler
from ufl.algorithms.map_integrands import map_integrand_dags

from ufl.classes import (ReferenceValue,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         Index)

from ufl.constantvalue import as_ufl, Identity
from ufl.tensors import as_tensor, as_vector

from ufl.finiteelement import (FiniteElement, EnrichedElement, VectorElement, MixedElement,
                               TensorProductElement, TensorProductVectorElement, TensorElement,
                               FacetElement, InteriorElement, BrokenElement, TraceElement)
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
        return [None]*shape[0]
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
        return [reshape_to_nested_list(components[n*i:n*(i+1)], shape[1:]) for i in range(shape[0])]

def apply_single_function_pullbacks(g):
    element = g.ufl_element()
    mapping = element.mapping()

    r = ReferenceValue(g)

    gsh = g.ufl_shape
    rsh = r.ufl_shape

    # Shortcut the "identity" case which includes Expression and Constant from dolfin that may be ill-formed without a domain (until we get that fixed)
    if mapping == "identity":
        assert rsh == gsh
        return r

    gsize = product(gsh)
    rsize = product(rsh)

    # Create some geometric objects for reuse
    domain = g.ufl_domain()
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)

    tdim = domain.topological_dimension()
    gdim = domain.geometric_dimension()

    # Create contravariant transform for reuse
    # (note that detJ is the _signed_ (pseudo-)determinant)
    transform_hdiv = (1.0/detJ) * J

    # Shortcut simple cases for a more efficient representation,
    # including directly Piola-mapped elements and mixed elements
    # of any combination of affinely mapped elements without symmetries
    if mapping == "symmetries":
        fcm = element.flattened_sub_element_mapping()
        assert gsize >= rsize
        assert len(fcm) == gsize
        assert sorted(set(fcm)) == sorted(range(rsize))
        g_components = [r[fcm[i]] for i in range(gsize)]
        g_components = reshape_to_nested_list(g_components, gsh)
        f = as_tensor(g_components)
        assert f.ufl_shape == g.ufl_shape
        return f
    elif mapping == "contravariant Piola":
        assert transform_hdiv.ufl_shape == (gsize, rsize)
        i, j = indices(2)
        f = as_vector(transform_hdiv[i, j]*r[j], i)
        #f = as_tensor(transform_hdiv[i, j]*r[k,j], (k,i)) # FIXME: Handle Vector(Piola) here?
        assert f.ufl_shape == g.ufl_shape
        return f
    elif mapping == "covariant Piola":
        assert Jinv.ufl_shape == (rsize, gsize)
        i, j = indices(2)
        f = as_vector(Jinv[j, i]*r[j], i)
        #f = as_tensor(Jinv[j, i]*r[k,j], (k,i)) # FIXME: Handle Vector(Piola) here?
        assert f.ufl_shape == g.ufl_shape
        return f


    # By placing components in a list and using as_vector at the end, we're
    # assuming below that both global function g and its reference value r
    # have vector shape, which is the case for most elements with the exceptions:
    # - TensorElements
    #   - All cases with scalar subelements and without symmetries are covered by the shortcut above
    #     (ONLY IF REFERENCE VALUE SHAPE PRESERVES TENSOR RANK)
    #   - All cases with scalar subelements and without symmetries are covered by the shortcut above
    # - VectorElements of vector-valued basic elements (FIXME)
    # - TensorElements with symmetries (FIXME)
    # - Tensor-valued FiniteElements (the new Regge elements)
    assert len(gsh) == 1
    assert len(rsh) == 1

    g_components = [None]*gsize
    gpos = 0
    rpos = 0
    for subelm in sub_elements_with_mappings(element):
        gm = product(subelm.value_shape())
        rm = product(subelm.reference_value_shape())

        mp = subelm.mapping()
        if mp == "identity":
            assert gm == rm
            for i in range(gm):
                g_components[gpos + i] = r[rpos + i]

        elif mp == "symmetries":
            """
            tensor_element.value_shape() == (2,2)
            tensor_element.reference_value_shape() == (3,)
            tensor_element.symmetry() == { (1,0): (0,1) }
            tensor_element.component_mapping() == { (0,0): 0, (0,1): 1, (1,0): 1, (1,1): 2 }
            tensor_element.flattened_component_mapping() == { 0: 0, 1: 1, 2: 1, 3: 2 }
            """
            fcm = subelm.flattened_sub_element_mapping()
            assert gm >= rm
            assert len(fcm) == gm
            assert sorted(set(fcm)) == sorted(range(rm))
            for i in range(gm):
                g_components[gpos + i] = r[rpos + fcm[i]]

        elif mp == "contravariant Piola":
            assert transform_hdiv.ufl_shape == (gm, rm)
            # Get reference value vector corresponding to this subelement:
            rv = as_vector([r[rpos+k] for k in range(rm)])
            # Apply transform with IndexSum over j for each row
            j = Index()
            for i in range(gm):
                g_components[gpos + i] = transform_hdiv[i, j]*rv[j]

        elif mp == "covariant Piola":
            assert Jinv.ufl_shape == (rm, gm)
            # Get reference value vector corresponding to this subelement:
            rv = as_vector([r[rpos+k] for k in range(rm)])
            # Apply transform with IndexSum over j for each row
            j = Index()
            for i in range(gm):
                g_components[gpos + i] = Jinv[j, i]*rv[j]

        else:
            error("Unknown subelement mapping type %s for element %s." % (mp, str(subelm)))

        gpos += gm
        rpos += rm

    # Wrap up components in a vector, must return same shape as input function g
    assert len(gsh) == 1
    f = as_vector(g_components)
    assert f.ufl_shape == g.ufl_shape
    return f


class FunctionPullbackApplier(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    @memoized_handler
    def form_argument(self, o):
        # Represent 0-derivatives of form arguments on reference element
        return apply_single_function_pullbacks(o)

def apply_function_pullbacks(expr):
    """Change representation of coefficients and arguments in expression
    by applying Piola mappings where applicable and representing all
    form arguments in reference value.

    @param expr:
        An Expr.
    """
    return map_integrand_dags(FunctionPullbackApplier(), expr)
