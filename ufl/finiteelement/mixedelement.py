"This module defines the UFL finite element classes."

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
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
#
# First added:  2008-03-03
# Last changed: 2012-10-18

from itertools import izip
from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.common import product, index_to_component, component_to_index, istr, EmptyDict
from ufl.domains import as_domain
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.finiteelement.finiteelement import FiniteElement


class MixedElement(FiniteElementBase):
    "A finite element composed of a nested hierarchy of mixed or simple elements"
    __slots__ = ("_sub_elements",)

    def __init__(self, *elements, **kwargs):
        "Create mixed finite element from given list of elements"

        # Un-nest arguments if we get a single argument with a list of elements
        if len(elements) == 1 and isinstance(elements[0], (tuple, list)):
            elements = elements[0]
        # Interpret nested tuples as sub-mixedelements recursively
        elements = [MixedElement(e) if isinstance(e, (tuple,list)) else e
                    for e in elements]
        self._sub_elements = elements

        # TODO: Figure out proper checks of domain consistency

        # FIXME: Do something if elements are defined on different regions
        domains = set(element.domain() for element in elements) - set((None,))
        top_domains = set(domain.top_domain() for domain in domains)
        if len(top_domains) == 0:
            domain = None
        elif len(top_domains) == 1:
            domain = list(top_domains)[0]
        else:
            # Require that all elements are defined on the same top-level domain
            # TODO: is this too strict? disallows mixed elements on different meshes
            error("Sub elements must live in the same top level domain.")

        # Check that domains have same geometric dimension
        if domain is not None:
            gdim = domain.geometric_dimension()
            ufl_assert(all(domain.geometric_dimension() == gdim for domain in domains),
                       "Sub elements must live in the same geometric dimension.")

        # Check that all elements use the same quadrature scheme
        # TODO: We can allow the scheme not to be defined.
        quad_scheme = elements[0].quadrature_scheme()
        ufl_assert(all(e.quadrature_scheme() == quad_scheme for e in elements),\
            "Quadrature scheme mismatch for sub elements of mixed element.")

        # Compute value shape
        value_size_sum = sum(product(s.value_shape()) for s in self._sub_elements)
        # Default value dimension: Treated simply as all subelement
        #                          values unpacked in a vector.
        value_shape = kwargs.get('value_shape', (value_size_sum,))
        # Validate value_shape
        if type(self) is MixedElement:
            # This is not valid for tensor elements with symmetries,
            # assume subclasses deal with their own validation
            ufl_assert(product(value_shape) == value_size_sum,
                "Provided value_shape doesn't match the total "\
                "value size of all subelements.")

        # Initialize element data
        degree = max(e.degree() for e in self._sub_elements)
        super(MixedElement, self).__init__("Mixed", domain, degree,
                                           quad_scheme, value_shape)

        # Cache repr string
        self._repr = "MixedElement(*%r, **{'value_shape': %r })" %\
            (self._sub_elements, self._value_shape)

    def reconstruct(self, **kwargs):
        """Construct a new MixedElement object with some
        properties replaced with new values."""
        elements = [e.reconstruct(**kwargs) for e in self._sub_elements]
        # Value shape cannot be changed, or at
        # least we have no valid use case for it.
        # Reconstructing an expression with a reconstructed
        # coefficient with a different value shape would
        # be way into undefined behaviour territory...
        ufl_assert("value_shape" not in kwargs,
                   "Cannot change value_shape in reconstruct.")
        return self.reconstruct_from_elements(*elements)

    def reconstruct_from_elements(self, *elements):
        "Reconstruct a mixed element from new subelements."
        if all(a == b for (a,b) in izip(elements, self._sub_elements)):
            return self
        ufl_assert(all(a.value_shape() == b.value_shape()
                       for (a,b) in izip(elements, self._sub_elements)),
            "Expecting new elements to have same value shape as old ones.")
        return MixedElement(*elements, value_shape=self.value_shape())

    def symmetry(self):
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1.
        A component is a tuple of one or more ints."""
        # Build symmetry map from symmetries of subelements
        sm = {}
        # Base index of the current subelement into mixed value
        j = 0
        for e in self._sub_elements:
            sh = e.value_shape()
            # Map symmetries of subelement into index space of this element
            for c0, c1 in e.symmetry().iteritems():
                j0 = component_to_index(c0, sh) + j
                j1 = component_to_index(c1, sh) + j
                sm[(j0,)] = (j1,)
            # Update base index for next element
            j += product(sh)
        ufl_assert(j == product(self.value_shape()),
                   "Size mismatch in symmetry algorithm.")
        return sm or EmptyDict

    def num_sub_elements(self):
        "Return number of sub elements."
        return len(self._sub_elements)

    def sub_elements(self):
        "Return list of sub elements."
        return self._sub_elements

    def num_mixed_sub_elements(self):
        "Return number of mixed sub elements."
        # FIXME: Use this where intended, for disambiguation
        #        w.r.t. different sub_elements meanings.
        return len(self._sub_elements)

    def mixed_sub_elements(self):
        "Return list of mixed sub elements."
        # FIXME: Use this where intended, for disambiguation
        #        w.r.t. different sub_elements meanings.
        return self._sub_elements

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative
        component index for a given component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)

        # Select between indexing modes
        if len(self.value_shape()) == 1:
            # Indexing into a long vector of flattened subelement shapes
            j, = i

            # Find subelement for this index
            for k, e in enumerate(self._sub_elements):
                sh = e.value_shape()
                si = product(sh)
                if j < si:
                    break
                j -= si
            ufl_assert(j >= 0, "Moved past last value component!")

            # Convert index into a shape tuple
            j = index_to_component(j, sh)
        else:
            # Indexing into a multidimensional tensor
            # where subelement index is first axis
            k = i[0]
            ufl_assert(k < len(self._sub_elements),
                       "Illegal component index (dimension %d)." % k)
            j = i[1:]
        return (k, j)

    def extract_component(self, i):
        """Recursively extract component index relative to a (simple) element
        and that element for given value component index"""
        k, j = self.extract_subelement_component(i)
        return self._sub_elements[k].extract_component(j)

    def is_cellwise_constant(self, component=None):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        if component is None:
            return all(e.is_cellwise_constant() for e in self.sub_elements())
        else:
            i, e = self.extract_component(component)
            return e.is_cellwise_constant()

    def domain(self, component=None):
        "Return the domain on which this element is defined."
        if component is None:
            return self._domain # FIXME: Remove this
        elif False and component is None:
            domains = [e.domain() for e in self.sub_elements()]
            domain = domains[0]
            for d in domains[1:]:
                domain = intersection(domain, d) # FIXME: Need intersection to make this work
            return domain
        else:
            i, e = self.extract_component(component)
            return e.domain()

    def regions(self):
        "Return the regions referenced by this element and its subelements."
        regions = set()
        for e in self.sub_elements():
            regions.update(e.regions())
        return sorted(regions)

    def degree(self, component=None):
        "Return polynomial degree of finite element"
        if component is None:
            return self._degree # from FiniteElementBase, computed as max of subelements in __init__
        else:
            i, e = self.extract_component(component)
            return e.degree()

    def __str__(self):
        "Format as string for pretty printing."
        tmp = ", ".join(str(element) for element in self._sub_elements)
        return "<Mixed element: (" + tmp + ")>"

    def shortstr(self):
        "Format as string for pretty printing."
        tmp = ", ".join(element.shortstr() for element in self._sub_elements)
        return "Mixed<" + tmp + ">"


class VectorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"

    def __init__(self, family, domain, degree, dim=None, quad_scheme=None,
                 form_degree=None):
        """
        Create vector element (repeated mixed element)

        *Arguments*
            family (string)
               The finite element family
            domain
               The geometric domain
            degree (int)
               The polynomial degree
            dim (int)
               The value dimension of the element (optional)
            quad_scheme
               The quadrature scheme (optional)
            form_degree (int)
               The form degree (FEEC notation, used when field is
               viewed as k-form)
        """
        if domain is not None:
            domain = as_domain(domain)

        # Set default size if not specified
        if dim is None:
            ufl_assert(domain is not None,
                       "Cannot infer vector dimension without a domain.")
            dim = domain.geometric_dimension()

        # Create mixed element from list of finite elements
        sub_element = FiniteElement(family, domain, degree, quad_scheme,
                                    form_degree)
        sub_elements = [sub_element]*dim

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Compute value shape
        shape = (dim,)
        value_shape = shape + sub_element.value_shape()

        # Initialize element data
        super(VectorElement, self).__init__(sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = degree
        self._sub_element = sub_element

        # Cache repr string
        self._repr = "VectorElement(%r, %r, %r, %d, %r)" % \
            (self._family, self._domain, self._degree,
             len(self._sub_elements), quad_scheme)

    def reconstruct(self, **kwargs):
        kwargs["family"] = kwargs.get("family", self.family())
        kwargs["domain"] = kwargs.get("domain", self.domain())
        kwargs["degree"] = kwargs.get("degree", self.degree())
        ufl_assert("dim" not in kwargs, "Cannot change dim in reconstruct.")
        kwargs["dim"] = len(self._sub_elements)
        kwargs["quad_scheme"] = kwargs.get("quad_scheme", self.quadrature_scheme())
        return VectorElement(**kwargs)

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s vector element of degree %s on a %s: %d x %s>" % \
               (self.family(), istr(self.degree()), self.domain(),
                len(self._sub_elements), self._sub_element)

    def shortstr(self):
        "Format as string for pretty printing."
        return "Vector<%d x %s>" % (len(self._sub_elements),
                                    self._sub_element.shortstr())

class TensorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    __slots__ = ("_sub_element", "_shape", "_symmetry", "_sub_element_mapping",)
    def __init__(self, family, domain, degree, shape=None,
                 symmetry=None, quad_scheme=None):
        "Create tensor element (repeated mixed element with optional symmetries)"
        if domain is not None:
            domain = as_domain(domain)

        # Set default shape if not specified
        if shape is None:
            ufl_assert(domain is not None,
                       "Cannot infer vector dimension without a domain.")
            dim = domain.geometric_dimension()
            shape = (dim, dim)

        # Construct default symmetry for matrix elements
        if symmetry == True:
            ufl_assert(len(shape) == 2 and shape[0] == shape[1],
                       "Cannot set automatic symmetry for non-square tensor.")
            symmetry = dict( ((i,j), (j,i)) for i in range(shape[0])
                             for j in range(shape[1]) if i > j )

        # Validate indices in symmetry dict
        if isinstance(symmetry, dict):
            for i,j in symmetry.iteritems():
                ufl_assert(len(i) == len(j),
                           "Non-matching length of symmetry index tuples.")
                for k in range(len(i)):
                    ufl_assert(i[k] >= 0 and j[k] >= 0 and
                               i[k] < shape[k] and j[k] < shape[k],
                               "Symmetry dimensions out of bounds.")
        else:
            ufl_assert(symmetry is None, "Expecting symmetry to be None, True, or dict.")

        # Compute all index combinations for given shape
        indices = compute_indices(shape)

        # Compute sub elements and mapping from indices
        # to sub elements, accounting for symmetry
        sub_element = FiniteElement(family, domain, degree, quad_scheme)
        sub_elements = []
        sub_element_mapping = {}
        for index in indices:
            if symmetry and index in symmetry:
                continue
            sub_element_mapping[index] = len(sub_elements)
            sub_elements += [sub_element]

        # Update mapping for symmetry
        for index in indices:
            if symmetry and index in symmetry:
                sub_element_mapping[index] = sub_element_mapping[symmetry[index]]

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Compute value shape
        value_shape = shape + sub_element.value_shape()

        # Initialize element data
        super(TensorElement, self).__init__(sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = degree
        self._sub_element = sub_element
        self._shape = shape
        self._symmetry = symmetry
        self._sub_element_mapping = sub_element_mapping

        # Cache repr string
        self._repr = "TensorElement(%r, %r, %r, %r, %r, %r)" % \
            (self._family, self._domain, self._degree, self._shape,
             self._symmetry, quad_scheme)

    def reconstruct(self, **kwargs):
        kwargs["family"] = kwargs.get("family", self.family())
        kwargs["domain"] = kwargs.get("domain",   self.domain())
        kwargs["degree"] = kwargs.get("degree", self.degree())

        ufl_assert("shape" not in kwargs, "Cannot change shape in reconstruct.")
        kwargs["shape"] = self.value_shape() # Must use same shape as self!

        # Not sure about symmetry, but no use case I can see
        ufl_assert("symmetry" not in kwargs, "Cannot change symmetry in reconstruct.")
        kwargs["symmetry"] = self.symmetry()

        kwargs["quad_scheme"] = kwargs.get("quad_scheme", self.quadrature_scheme())
        return TensorElement(**kwargs)

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative
        component index for a given component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)

        i = self.symmetry().get(i, i)
        l = len(self._shape)
        ii = i[:l]
        jj = i[l:]
        ufl_assert(ii in self._sub_element_mapping,
                   "Illegal component index %s." % repr(i))
        k = self._sub_element_mapping[ii]
        return (k, jj)

    def symmetry(self):
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return self._symmetry or EmptyDict

    def __str__(self):
        "Format as string for pretty printing."
        sym = ""
        if isinstance(self._symmetry, dict):
            tmp = ", ".join("%s -> %s" % (a,b) for (a,b) in self._symmetry.iteritems())
            sym = " with symmetries (%s)" % tmp
        elif self._symmetry:
            sym = " with symmetry"
        return "<%s tensor element of degree %s and shape %s on a %s%s>" % \
            (self.family(), istr(self.degree()), self.value_shape(), self.domain(), sym)

    def shortstr(self):
        "Format as string for pretty printing."
        sym = ""
        if isinstance(self._symmetry, dict):
            tmp = ", ".join("%s -> %s" % (a,b) for (a,b) in self._symmetry.iteritems())
            sym = " with symmetries (%s)" % tmp
        elif self._symmetry:
            sym = " with symmetry"
        return "Tensor<%s x %s%s>" % (self.value_shape(),
                                      self._sub_element.shortstr(), sym)
