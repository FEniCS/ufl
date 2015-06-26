"""Algorithm for replacing gradients in an expression with reference gradients and coordinate mappings."""

# Copyright (C) 2008-2015 Martin Sandve Alnes
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
                         CellOrientation, FacetOrientation)

from ufl.constantvalue import as_ufl, Identity
from ufl.tensors import as_tensor, as_vector

from ufl.finiteelement import (FiniteElement, EnrichedElement, VectorElement, MixedElement,
                               OuterProductElement, OuterProductVectorElement, TensorElement,
                               FacetElement, InteriorElement, BrokenElement, TraceElement)

def contravariant_hdiv_mapping(domain):
    "Return the contravariant H(div) mapping matrix (1/det J) * J"
    tdim = domain.topological_dimension()
    gdim = domain.geometric_dimension()
    ufl_assert(tdim > 1, "Cannot have Piola-mapped element in 1D")
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    piola_trans = (1.0/detJ) * J
    # Only insert symbolic CellOrientation if tdim != gdim
    if tdim != gdim:
        piola_trans = CellOrientation(domain) * piola_trans
    return piola_trans

def covariant_hcurl_mapping(domain):
    "Return the covariant H(curl) mapping matrix JinvT."
    tdim = domain.topological_dimension()
    gdim = domain.geometric_dimension()
    ufl_assert(tdim > 1, "Cannot have Piola-mapped element in 1D")
    Jinv = JacobianInverse(domain)
    # Using indexing for the transpose
    i, j = indices(2)
    JinvT = as_tensor(Jinv[i, j], (j, i))
    piola_trans = JinvT
    return piola_trans

def combine_transforms(element, subtransforms):
    # "current" position to insert to
    pos = 0
    # number of columns
    width = element.reference_value_shape()[0]

    new_tensor = []
    for ii, subelt in enumerate(element.sub_elements()):
        if len(subelt.value_shape()) == 0:
            # scalar-valued
            new_row = [0,]*width
            new_row[pos] = subtransforms[ii]
            new_tensor.append(new_row)
            pos += 1
        elif len(subelt.value_shape()) == 1:
            # vector-valued
            local_width = subelt.reference_value_shape()[0]
            for jj in range(subelt.value_shape()[0]):
                new_row = [0,]*width
                new_row[pos:pos+local_width] = subtransforms[ii][jj,:]
                new_tensor.append(new_row)
            pos += local_width
        else:
            error("can't handle %s in a MixedElement", str(subelt))
    return as_tensor(new_tensor)

def build_pullback_transform(domain, element):
    mapping = element.mapping()
    if mapping == "identity":
        return as_ufl(1.0)
    elif mapping == "contravariant Piola":
        return contravariant_hdiv_mapping(domain)
    elif mapping == "covariant Piola":
        return covariant_hcurl_mapping(domain)
    elif isinstance(element, MixedElement):
        subtransforms = [build_pullback_transform(domain, elm) for elm in element.sub_elements()]
        return combine_transforms(subtransforms)
    else:
        error("Mapping type %s not handled for element type %s" % (mapping, element.__class__.__name__))

class ElementMappingApplier(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, t):
        return t

    @memoized_handler
    def form_argument(self, o):
        # Represent 0-derivatives of form arguments on reference element

        element = o.element()
        local_value = ReferenceValue(o)
        domain = o.domain()

        # Split into a separate function to allow MixedElement recursion
        transform = build_pullback_transform(domain, element)

        r = len(transform.ufl_shape)
        if r == 0:
            return transform*local_value
        elif r == 2:
            i, j = indices(2)
            return as_vector(transform[i, j] * local_value[j], i)
        else:
            error("Unknown transform %s", str(transform))


def apply_element_mappings(expr):
    """Change representation of coefficients and arguments in expression
    by applying Piola mappings where applicable and representing all
    form arguments in reference value.

    @param expr:
        An Expr.
    """
    return map_integrand_dags(ElementMappingApplier(), expr)
