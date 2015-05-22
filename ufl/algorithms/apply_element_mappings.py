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
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_integrand_dags

from ufl.classes import (ReferenceValue,
                         Jacobian, JacobianInverse, JacobianDeterminant,
                         CellOrientation, FacetOrientation)

from ufl.constantvalue import as_ufl, Identity
from ufl.tensors import as_tensor, as_vector

from ufl.finiteelement import (FiniteElement, EnrichedElement, VectorElement, MixedElement,
                               OuterProductElement, OuterProductVectorElement, TensorElement,
                               FacetElement, InteriorElement, BrokenElement, TraceElement)

def _reference_value_helper(domain, element):
    element_types = (FiniteElement, EnrichedElement,
                     OuterProductElement, TensorElement,
                     FacetElement, InteriorElement,
                     BrokenElement, TraceElement)
    if isinstance(element, element_types):
        mapping = element.mapping()

        if mapping == "identity":
            return as_ufl(1.0)

        elif mapping == "contravariant Piola":
            ufl_assert(domain.topological_dimension() >= 2,
                       "Cannot have Piola-mapped element in 1D")

            # contravariant_hdiv_mapping = (1/det J) * J * PullbackOf(o)
            J = Jacobian(domain)
            detJ = JacobianDeterminant(domain)
            # Only insert symbolic CellOrientation if tdim != gdim
            if domain.topological_dimension() == domain.geometric_dimension():
                piola_trans = (1/detJ) * J
            else:
                piola_trans = CellOrientation(domain) * (1/detJ) * J
            return piola_trans

        elif mapping == "covariant Piola":
            ufl_assert(domain.topological_dimension() >= 2,
                       "Cannot have Piola-mapped element in 1D")

            # covariant_hcurl_mapping = JinvT * PullbackOf(o)
            Jinv = JacobianInverse(domain)
            i, j = indices(2)
            JinvT = as_tensor(Jinv[i, j], (j, i))
            piola_trans = JinvT
            return piola_trans

        else:
            error("Mapping type %s not handled" % mapping)

    elif isinstance(element, (VectorElement, OuterProductVectorElement)):
        # Allow VectorElement of CG/DG (scalar-valued), throw error
        # on anything else (can be supported at a later date, if needed)
        mapping = element.mapping()
        if mapping == "identity" and len(element.value_shape()) == 1:
            return Identity(element.value_shape()[0])
        else:
            error("Don't know how to handle %s", str(element))

    elif isinstance(element, MixedElement):
        temp = [_reference_value_helper(domain, foo) for foo in element.sub_elements()]
        # "current" position to insert to
        hh = 0
        # number of columns
        width = element.reference_value_shape()[0]

        new_tensor = []
        for ii, subelt in enumerate(element.sub_elements()):
            if len(subelt.value_shape()) == 0:
                # scalar-valued
                new_row = [0,]*width
                new_row[hh] = temp[ii]
                new_tensor.append(new_row)
                hh += 1
            elif len(subelt.value_shape()) == 1:
                # vector-valued
                local_width = subelt.reference_value_shape()[0]
                for jj in range(subelt.value_shape()[0]):
                    new_row = [0,]*width
                    new_row[hh:hh+local_width] = temp[ii][jj,:]
                    new_tensor.append(new_row)
                hh += local_width
            else:
                error("can't handle %s in a MixedElement", str(subelt))

        return as_tensor(new_tensor)

    else:
        error("Unknown element %s", str(element))


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
        transform = _reference_value_helper(domain, element)

        r = len(transform.shape())
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
        An Expr or Form.
    """
    return map_integrand_dags(ElementMappingApplier(), expr)
