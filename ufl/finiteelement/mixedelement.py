# -*- coding: utf-8 -*-
"This module defines the UFL finite element classes."

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
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Anders Logg 2014

from itertools import chain

from six import iteritems
from six.moves import zip
from six.moves import xrange as range

from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.utils.sequences import product
from ufl.utils.formatting import istr
from ufl.utils.dicts import EmptyDict
from ufl.utils.indexflattening import flatten_multiindex, unflatten_index, shape_to_strides
from ufl.cell import as_cell
from ufl.log import info_blue, warning, warning_blue, error

from ufl.finiteelement.finiteelementbase import FiniteElementBase
from ufl.finiteelement.finiteelement import FiniteElement


class MixedElement(FiniteElementBase):
    "A finite element composed of a nested hierarchy of mixed or simple elements"
    __slots__ = ("_sub_elements", "_cells")

    def __init__(self, *elements, **kwargs):
        "Create mixed finite element from given list of elements"

        if type(self) is MixedElement:
            ufl_assert(not kwargs, "Not expecting keyword arguments to MixedElement constructor.")

        # Un-nest arguments if we get a single argument with a list of elements
        if len(elements) == 1 and isinstance(elements[0], (tuple, list)):
            elements = elements[0]
        # Interpret nested tuples as sub-mixedelements recursively
        elements = [MixedElement(e) if isinstance(e, (tuple, list)) else e
                    for e in elements]
        self._sub_elements = elements

        # Pick the first cell, for now all should be equal
        cells = tuple(sorted(set(element.cell() for element in elements) - set([None])))
        self._cells = cells
        if cells:
            cell = cells[0]
            # Require that all elements are defined on the same cell
            ufl_assert(all(c == cell for c in cells[1:]),
                       "Sub elements must live on the same cell.")
        else:
            cell = None

        # Check that all elements use the same quadrature scheme
        # TODO: We can allow the scheme not to be defined.
        quad_scheme = elements[0].quadrature_scheme()
        ufl_assert(all(e.quadrature_scheme() == quad_scheme for e in elements),
            "Quadrature scheme mismatch for sub elements of mixed element.")

        # Compute value sizes in global and reference configurations
        value_size_sum = sum(product(s.value_shape()) for s in self._sub_elements)
        reference_value_size_sum = sum(product(s.reference_value_shape()) for s in self._sub_elements)

        # Default value shape: Treated simply as all subelement values unpacked in a vector.
        value_shape = kwargs.get('value_shape', (value_size_sum,))

        # Default reference value shape: Treated simply as all subelement reference values unpacked in a vector.
        reference_value_shape = kwargs.get('reference_value_shape', (reference_value_size_sum,))

        # Validate value_shape (deliberately not for subclasses VectorElement and TensorElement)
        if type(self) is MixedElement:
            # This is not valid for tensor elements with symmetries,
            # assume subclasses deal with their own validation
            ufl_assert(product(value_shape) == value_size_sum,
                "Provided value_shape doesn't match the total "
                "value size of all subelements.")

        # Initialize element data
        degrees = { e.degree() for e in self._sub_elements } - { None }
        degree = max(degrees) if degrees else None
        FiniteElementBase.__init__(self, "Mixed", cell, degree, quad_scheme,
                                   value_shape, reference_value_shape)

        # Cache repr string
        if type(self) is MixedElement:
            self._repr = "MixedElement(%s)" % ", ".join(repr(e) for e in self._sub_elements)

    def reconstruct_from_elements(self, *elements):
        "Reconstruct a mixed element from new subelements."
        if all(a == b for (a, b) in zip(elements, self._sub_elements)):
            return self
        return MixedElement(*elements)

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
            st = shape_to_strides(sh)
            # Map symmetries of subelement into index space of this element
            for c0, c1 in iteritems(e.symmetry()):
                j0 = flatten_multiindex(c0, st) + j
                j1 = flatten_multiindex(c1, st) + j
                sm[(j0,)] = (j1,)
            # Update base index for next element
            j += product(sh)
        ufl_assert(j == product(self.value_shape()),
                   "Size mismatch in symmetry algorithm.")
        return sm or EmptyDict

    def mapping(self):
        if all(e.mapping() == "identity" for e in self._sub_elements):
            return "identity"
        else:
            return "undefined"

    def num_sub_elements(self):
        "Return number of sub elements."
        return len(self._sub_elements)

    def sub_elements(self):
        "Return list of sub elements."
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
            for sub_element_index, e in enumerate(self._sub_elements):
                sh = e.value_shape()
                si = product(sh)
                if j < si:
                    break
                j -= si
            ufl_assert(j >= 0, "Moved past last value component!")

            # Convert index into a shape tuple
            st = shape_to_strides(sh)
            component = unflatten_index(j, st)
        else:
            # Indexing into a multidimensional tensor
            # where subelement index is first axis
            sub_element_index = i[0]
            ufl_assert(sub_element_index < len(self._sub_elements),
                       "Illegal component index (dimension %d)." % sub_element_index)
            component = i[1:]
        return (sub_element_index, component)

    def extract_component(self, i):
        """Recursively extract component index relative to a (simple) element
        and that element for given value component index"""
        sub_element_index, component = self.extract_subelement_component(i)
        return self._sub_elements[sub_element_index].extract_component(component)

    def extract_subelement_reference_component(self, i):
        """Extract direct subelement index and subelement relative
        reference_component index for a given reference_component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)

        # Select between indexing modes
        assert len(self.reference_value_shape()) == 1
        # Indexing into a long vector of flattened subelement shapes
        j, = i

        # Find subelement for this index
        for sub_element_index, e in enumerate(self._sub_elements):
            sh = e.reference_value_shape()
            si = product(sh)
            if j < si:
                break
            j -= si
        ufl_assert(j >= 0, "Moved past last value reference_component!")

        # Convert index into a shape tuple
        st = shape_to_strides(sh)
        reference_component = unflatten_index(j, st)
        return (sub_element_index, reference_component)

    def extract_reference_component(self, i):
        """Recursively extract reference_component index relative to a (simple) element
        and that element for given value reference_component index"""
        sub_element_index, reference_component = self.extract_subelement_reference_component(i)
        return self._sub_elements[sub_element_index].extract_reference_component(reference_component)

    def is_cellwise_constant(self, component=None):
        """Return whether the basis functions of this
        element is spatially constant over each cell."""
        if component is None:
            return all(e.is_cellwise_constant() for e in self.sub_elements())
        else:
            i, e = self.extract_component(component)
            return e.is_cellwise_constant()

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

    def __new__(cls, family, cell, degree, dim=None,
                form_degree=None, quad_scheme=None):
        """Intercepts construction, such that it returns an
        OuterProductVectorElement when FiniteElement returns an
        OuterProductElement.
        """
        # Create mixed element from list of finite elements
        sub_element = FiniteElement(family, cell, degree,
                                    form_degree=form_degree,
                                    quad_scheme=quad_scheme)

        from ufl.finiteelement.outerproductelement import OuterProductElement
        from ufl.finiteelement.outerproductelement import OuterProductVectorElement
        if isinstance(sub_element, OuterProductElement):
            return OuterProductVectorElement(sub_element, dim=dim)

        return super(VectorElement, cls).__new__(cls)

    def __init__(self, family, cell, degree, dim=None,
                 form_degree=None, quad_scheme=None):
        """
        Create vector element (repeated mixed element)

        *Arguments*
            family (string)
               The finite element family
            cell
               The geometric cell
            degree (int)
               The polynomial degree
            dim (int)
               The value dimension of the element (optional)
            form_degree (int)
               The form degree (FEEC notation, used when field is
               viewed as k-form)
            quad_scheme
               The quadrature scheme (optional)
        """
        if cell is not None:
            cell = as_cell(cell)

        # Set default size if not specified
        if dim is None:
            ufl_assert(cell is not None,
                       "Cannot infer vector dimension without a cell.")
            dim = cell.geometric_dimension()

        # Create sub element
        sub_element = FiniteElement(family, cell, degree,
                                    form_degree=form_degree,
                                    quad_scheme=quad_scheme)

        # Create list of sub elements for mixed element constructor
        sub_elements = [sub_element]*dim

        # Compute value shapes
        value_shape = (dim,) + sub_element.value_shape()
        reference_value_shape = (dim,) + sub_element.reference_value_shape()

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape, reference_value_shape=reference_value_shape)
        # FIXME: Storing this here is strange, isn't that handled by subclass?
        self._family = sub_element.family()
        self._degree = degree
        self._sub_element = sub_element
        self._form_degree = form_degree # Storing for signature_data, not sure if it's needed

        # Cache repr string
        qs = self.quadrature_scheme()
        quad_str = "" if qs is None else ", quad_scheme=%r" % (qs,)
        self._repr = ("VectorElement(%r, %r, %r, dim=%d%s)" %
            (self._family, self.cell(), self._degree,
             len(self._sub_elements), quad_str))

    def __str__(self):
        "Format as string for pretty printing."
        return ("<%s vector element of degree %s on a %s: %d x %s>" %
               (self.family(), istr(self.degree()), self.cell(),
                len(self._sub_elements), self._sub_element))

    def shortstr(self):
        "Format as string for pretty printing."
        return "Vector<%d x %s>" % (len(self._sub_elements),
                                    self._sub_element.shortstr())


class TensorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    __slots__ = ("_sub_element", "_shape", "_symmetry",
                 "_sub_element_mapping", "_flattened_sub_element_mapping",
                 "_mapping")

    def __new__(cls, family, cell, degree, shape=None,
                symmetry=None, quad_scheme=None):
        """Intercepts construction, such that it returns an
        OuterProductTensorElement when FiniteElement returns an
        OuterProductElement.
        """
        # Compute sub element
        sub_element = FiniteElement(family, cell, degree, quad_scheme)

        from ufl.finiteelement.outerproductelement import OuterProductElement
        from ufl.finiteelement.outerproductelement import OuterProductTensorElement
        if isinstance(sub_element, OuterProductElement):
            return OuterProductTensorElement(sub_element, shape=shape, symmetry=symmetry)

        return super(TensorElement, cls).__new__(cls)

    def __init__(self, family, cell, degree, shape=None,
                 symmetry=None, quad_scheme=None):
        "Create tensor element (repeated mixed element with optional symmetries)"
        # Create scalar sub element
        sub_element = FiniteElement(family, cell, degree, quad_scheme)
        ufl_assert(sub_element.value_shape() == (),
                   "Expecting only scalar valued subelement for TensorElement.")

        shape, symmetry, sub_elements, sub_element_mapping, flattened_sub_element_mapping, \
          reference_value_shape, mapping = _tensor_sub_elements(sub_element, shape, symmetry)

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=shape,
                              reference_value_shape=reference_value_shape)
        self._family = sub_element.family()
        self._degree = degree
        self._sub_element = sub_element
        self._shape = shape
        self._symmetry = symmetry
        self._sub_element_mapping = sub_element_mapping
        self._flattened_sub_element_mapping = flattened_sub_element_mapping
        self._mapping = mapping

        # Cache repr string
        qs = self.quadrature_scheme()
        quad_str = "" if qs is None else ", quad_scheme=%r" % (qs,)
        self._repr = ("TensorElement(%r, %r, %r, shape=%r, symmetry=%r%s)" %
            (self._family, self.cell(), self._degree, self._shape,
             self._symmetry, quad_str))

    def mapping(self):
        if self._symmetry:
            return "symmetries"
        else:
            return "identity"

    def flattened_sub_element_mapping(self):
        return self._flattened_sub_element_mapping

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
        return self._symmetry

    def __str__(self):
        "Format as string for pretty printing."
        if self._symmetry:
            tmp = ", ".join("%s -> %s" % (a, b) for (a, b) in iteritems(self._symmetry))
            sym = " with symmetries (%s)" % tmp
        else:
            sym = ""
        return ("<%s tensor element of degree %s and shape %s on a %s%s>" %
            (self.family(), istr(self.degree()), self.value_shape(), self.cell(), sym))

    def shortstr(self):
        "Format as string for pretty printing."
        if self._symmetry:
            tmp = ", ".join("%s -> %s" % (a, b) for (a, b) in iteritems(self._symmetry))
            sym = " with symmetries (%s)" % tmp
        else:
            sym = ""
        return "Tensor<%s x %s%s>" % (self.value_shape(),
                                      self._sub_element.shortstr(), sym)


def _tensor_sub_elements(sub_element, shape, symmetry):
    # Set default shape if not specified
    if shape is None:
        ufl_assert(sub_element.cell() is not None,
                   "Cannot infer tensor shape without a cell.")
        dim = sub_element.cell().geometric_dimension()
        shape = (dim, dim)

    if symmetry is None:
        symmetry = EmptyDict
    elif symmetry is True:
        # Construct default symmetry dict for matrix elements
        ufl_assert(len(shape) == 2 and shape[0] == shape[1],
                   "Cannot set automatic symmetry for non-square tensor.")
        symmetry = dict( ((i, j), (j, i)) for i in range(shape[0])
                         for j in range(shape[1]) if i > j )
    else:
        ufl_assert(isinstance(symmetry, dict), "Expecting symmetry to be None (unset), True, or dict.")

    # Validate indices in symmetry dict
    for i, j in iteritems(symmetry):
        ufl_assert(len(i) == len(j),
                   "Non-matching length of symmetry index tuples.")
        for k in range(len(i)):
            ufl_assert(i[k] >= 0 and j[k] >= 0 and
                       i[k] < shape[k] and j[k] < shape[k],
                       "Symmetry dimensions out of bounds.")

    # Compute all index combinations for given shape
    indices = compute_indices(shape)

    # Compute mapping from indices to sub element number, accounting for symmetry
    sub_elements = []
    sub_element_mapping = {}
    for index in indices:
        if index in symmetry:
            continue
        sub_element_mapping[index] = len(sub_elements)
        sub_elements += [sub_element]

    # Update mapping for symmetry
    for index in indices:
        if index in symmetry:
            sub_element_mapping[index] = sub_element_mapping[symmetry[index]]
    flattened_sub_element_mapping = [sub_element_mapping[index] for i, index in enumerate(indices)]

    # Compute reference value shape based on symmetries
    if symmetry:
        # Flatten and subtract symmetries
        reference_value_shape = (product(shape)-len(symmetry),)
        mapping = "symmetries"
    else:
        # Do not flatten if there are no symmetries
        reference_value_shape = shape
        mapping = "identity"


    return shape, symmetry, sub_elements, sub_element_mapping, \
      flattened_sub_element_mapping, reference_value_shape, mapping
