"""This module defines the UFL finite element classes."""
# Copyright (C) 2008-2016 Martin Sandve Alnæs
#
# This file is part of UFL (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes 2010, 2012
# Modified by Anders Logg 2014
# Modified by Massimiliano Leoni, 2016

import numpy as np

from ufl.cell import as_cell
from ufl.legacy.finiteelement import FiniteElement
from ufl.legacy.finiteelementbase import FiniteElementBase
from ufl.permutation import compute_indices
from ufl.pull_back import MixedPullBack, SymmetricPullBack
from ufl.utils.indexflattening import flatten_multiindex, shape_to_strides, unflatten_index
from ufl.utils.sequences import max_degree, product


class MixedElement(FiniteElementBase):
    """A finite element composed of a nested hierarchy of mixed or simple elements."""
    __slots__ = ("_sub_elements", "_cells")

    def __init__(self, *elements, **kwargs):
        """Create mixed finite element from given list of elements."""
        if type(self) is MixedElement:
            if kwargs:
                raise ValueError("Not expecting keyword arguments to MixedElement constructor.")

        # Un-nest arguments if we get a single argument with a list of elements
        if len(elements) == 1 and isinstance(elements[0], (tuple, list)):
            elements = elements[0]
        # Interpret nested tuples as sub-mixedelements recursively
        elements = [MixedElement(e) if isinstance(e, (tuple, list)) else e
                    for e in elements]
        self._sub_elements = elements

        # Pick the first cell, for now all should be equal
        cells = tuple(sorted(set(element.cell for element in elements) - set([None])))
        self._cells = cells
        if cells:
            cell = cells[0]
            # Require that all elements are defined on the same cell
            if not all(c == cell for c in cells[1:]):
                raise ValueError("Sub elements must live on the same cell.")
        else:
            cell = None

        # Check that all elements use the same quadrature scheme TODO:
        # We can allow the scheme not to be defined.
        if len(elements) == 0:
            quad_scheme = None
        else:
            quad_scheme = elements[0].quadrature_scheme()
            if not all(e.quadrature_scheme() == quad_scheme for e in elements):
                raise ValueError("Quadrature scheme mismatch for sub elements of mixed element.")

        # Compute value sizes in global and reference configurations
        value_size_sum = sum(product(s.value_shape) for s in self._sub_elements)
        reference_value_size_sum = sum(product(s.reference_value_shape) for s in self._sub_elements)

        # Default value shape: Treated simply as all subelement values
        # unpacked in a vector.
        value_shape = kwargs.get('value_shape', (value_size_sum,))

        # Default reference value shape: Treated simply as all
        # subelement reference values unpacked in a vector.
        reference_value_shape = kwargs.get('reference_value_shape', (reference_value_size_sum,))

        # Validate value_shape (deliberately not for subclasses
        # VectorElement and TensorElement)
        if type(self) is MixedElement:
            # This is not valid for tensor elements with symmetries,
            # assume subclasses deal with their own validation
            if product(value_shape) != value_size_sum:
                raise ValueError("Provided value_shape doesn't match the "
                                 "total value size of all subelements.")

        # Initialize element data
        degrees = {e.degree() for e in self._sub_elements} - {None}
        degree = max_degree(degrees) if degrees else None
        FiniteElementBase.__init__(self, "Mixed", cell, degree, quad_scheme,
                                   value_shape, reference_value_shape)

    def __repr__(self):
        """Doc."""
        return "MixedElement(" + ", ".join(repr(e) for e in self._sub_elements) + ")"

    def _is_linear(self):
        """Doc."""
        return all(i._is_linear() for i in self._sub_elements)

    def reconstruct_from_elements(self, *elements):
        """Reconstruct a mixed element from new subelements."""
        if all(a == b for (a, b) in zip(elements, self._sub_elements)):
            return self
        return MixedElement(*elements)

    def symmetry(self):
        r"""Return the symmetry dict, which is a mapping :math:`c_0 \\to c_1`.

        meaning that component :math:`c_0` is represented by component
        :math:`c_1`.
        A component is a tuple of one or more ints.
        """
        # Build symmetry map from symmetries of subelements
        sm = {}
        # Base index of the current subelement into mixed value
        j = 0
        for e in self._sub_elements:
            sh = e.value_shape
            st = shape_to_strides(sh)
            # Map symmetries of subelement into index space of this
            # element
            for c0, c1 in e.symmetry().items():
                j0 = flatten_multiindex(c0, st) + j
                j1 = flatten_multiindex(c1, st) + j
                sm[(j0,)] = (j1,)
            # Update base index for next element
            j += product(sh)
        if j != product(self.value_shape):
            raise ValueError("Size mismatch in symmetry algorithm.")
        return sm or {}

    @property
    def sobolev_space(self):
        """Doc."""
        return max(e.sobolev_space() for e in self._sub_elements)

    def mapping(self):
        """Doc."""
        if all(e.mapping() == "identity" for e in self._sub_elements):
            return "identity"
        else:
            return "undefined"

    @property
    def num_sub_elements(self):
        """Return number of sub elements."""
        return len(self._sub_elements)

    @property
    def sub_elements(self):
        """Return list of sub elements."""
        return self._sub_elements

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative.

        component index for a given component index.
        """
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)

        # Select between indexing modes
        if len(self.value_shape) == 1:
            # Indexing into a long vector of flattened subelement
            # shapes
            j, = i

            # Find subelement for this index
            for sub_element_index, e in enumerate(self._sub_elements):
                sh = e.value_shape
                si = product(sh)
                if j < si:
                    break
                j -= si
            if j < 0:
                raise ValueError("Moved past last value component!")

            # Convert index into a shape tuple
            st = shape_to_strides(sh)
            component = unflatten_index(j, st)
        else:
            # Indexing into a multidimensional tensor where subelement
            # index is first axis
            sub_element_index = i[0]
            if sub_element_index >= len(self._sub_elements):
                raise ValueError(f"Illegal component index (dimension {sub_element_index}).")
            component = i[1:]
        return (sub_element_index, component)

    def extract_component(self, i):
        """Recursively extract component index relative to a (simple) element.

        and that element for given value component index.
        """
        sub_element_index, component = self.extract_subelement_component(i)
        return self._sub_elements[sub_element_index].extract_component(component)

    def extract_subelement_reference_component(self, i):
        """Extract direct subelement index and subelement relative.

        reference_component index for a given reference_component index.
        """
        if isinstance(i, int):
            i = (i,)
        self._check_reference_component(i)

        # Select between indexing modes
        assert len(self.reference_value_shape) == 1
        # Indexing into a long vector of flattened subelement shapes
        j, = i

        # Find subelement for this index
        for sub_element_index, e in enumerate(self._sub_elements):
            sh = e.reference_value_shape
            si = product(sh)
            if j < si:
                break
            j -= si
        if j < 0:
            raise ValueError("Moved past last value reference_component!")

        # Convert index into a shape tuple
        st = shape_to_strides(sh)
        reference_component = unflatten_index(j, st)
        return (sub_element_index, reference_component)

    def extract_reference_component(self, i):
        """Recursively extract reference_component index relative to a (simple) element.

        and that element for given value reference_component index.
        """
        sub_element_index, reference_component = self.extract_subelement_reference_component(i)
        return self._sub_elements[sub_element_index].extract_reference_component(reference_component)

    def is_cellwise_constant(self, component=None):
        """Return whether the basis functions of this element is spatially constant over each cell."""
        if component is None:
            return all(e.is_cellwise_constant() for e in self.sub_elements())
        else:
            i, e = self.extract_component(component)
            return e.is_cellwise_constant()

    def degree(self, component=None):
        """Return polynomial degree of finite element."""
        if component is None:
            return self._degree  # from FiniteElementBase, computed as max of subelements in __init__
        else:
            i, e = self.extract_component(component)
            return e.degree()

    def reconstruct(self, **kwargs):
        """Doc."""
        return MixedElement(*[e.reconstruct(**kwargs) for e in self.sub_elements()])

    def variant(self):
        """Doc."""
        try:
            variant, = {e.variant() for e in self.sub_elements()}
            return variant
        except ValueError:
            return None

    def __str__(self):
        """Format as string for pretty printing."""
        tmp = ", ".join(str(element) for element in self._sub_elements)
        return "<Mixed element: (" + tmp + ")>"

    def shortstr(self):
        """Format as string for pretty printing."""
        tmp = ", ".join(element.shortstr() for element in self._sub_elements)
        return "Mixed<" + tmp + ">"

    @property
    def pull_back(self):
        """Get the pull back."""
        return MixedPullBack(self)


class VectorElement(MixedElement):
    """A special case of a mixed finite element where all elements are equal."""

    __slots__ = ("_repr", "_mapping", "_sub_element")

    def __init__(self, family, cell=None, degree=None, dim=None,
                 form_degree=None, quad_scheme=None, variant=None):
        """Create vector element (repeated mixed element)."""
        if isinstance(family, FiniteElementBase):
            sub_element = family
            cell = sub_element.cell
            variant = sub_element.variant()
        else:
            if cell is not None:
                cell = as_cell(cell)
            # Create sub element
            sub_element = FiniteElement(family, cell, degree,
                                        form_degree=form_degree,
                                        quad_scheme=quad_scheme,
                                        variant=variant)

        # Set default size if not specified
        if dim is None:
            if cell is None:
                raise ValueError("Cannot infer vector dimension without a cell.")
            dim = cell.geometric_dimension()

        self._mapping = sub_element.mapping()
        # Create list of sub elements for mixed element constructor
        sub_elements = [sub_element] * dim

        # Compute value shapes
        value_shape = (dim,) + sub_element.value_shape
        reference_value_shape = (dim,) + sub_element.reference_value_shape

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape,
                              reference_value_shape=reference_value_shape)

        FiniteElementBase.__init__(self, sub_element.family(), sub_element.cell, sub_element.degree(),
                                   sub_element.quadrature_scheme(), value_shape, reference_value_shape)

        self._sub_element = sub_element

        if variant is None:
            var_str = ""
        else:
            var_str = ", variant='" + variant + "'"

        # Cache repr string
        self._repr = f"VectorElement({repr(sub_element)}, dim={dim}{var_str})"

    def __repr__(self):
        """Doc."""
        return self._repr

    def reconstruct(self, **kwargs):
        """Doc."""
        sub_element = self._sub_element.reconstruct(**kwargs)
        return VectorElement(sub_element, dim=len(self.sub_elements()))

    def variant(self):
        """Return the variant used to initialise the element."""
        return self._sub_element.variant()

    def mapping(self):
        """Doc."""
        return self._mapping

    def __str__(self):
        """Format as string for pretty printing."""
        return ("<vector element with %d components of %s>" %
                (len(self._sub_elements), self._sub_element))

    def shortstr(self):
        """Format as string for pretty printing."""
        return "Vector<%d x %s>" % (len(self._sub_elements),
                                    self._sub_element.shortstr())


class TensorElement(MixedElement):
    """A special case of a mixed finite element where all elements are equal."""
    __slots__ = ("_sub_element", "_shape", "_symmetry",
                 "_sub_element_mapping",
                 "_flattened_sub_element_mapping",
                 "_mapping", "_repr")

    def __init__(self, family, cell=None, degree=None, shape=None,
                 symmetry=None, quad_scheme=None, variant=None):
        """Create tensor element (repeated mixed element with optional symmetries)."""
        if isinstance(family, FiniteElementBase):
            sub_element = family
            cell = sub_element.cell
            variant = sub_element.variant()
        else:
            if cell is not None:
                cell = as_cell(cell)
            # Create scalar sub element
            sub_element = FiniteElement(family, cell, degree, quad_scheme=quad_scheme,
                                        variant=variant)

        # Set default shape if not specified
        if shape is None:
            if cell is None:
                raise ValueError("Cannot infer tensor shape without a cell.")
            dim = cell.geometric_dimension()
            shape = (dim, dim)

        if symmetry is None:
            symmetry = {}
        elif symmetry is True:
            # Construct default symmetry dict for matrix elements
            if not (len(shape) == 2 and shape[0] == shape[1]):
                raise ValueError("Cannot set automatic symmetry for non-square tensor.")
            symmetry = dict(((i, j), (j, i)) for i in range(shape[0])
                            for j in range(shape[1]) if i > j)
        else:
            if not isinstance(symmetry, dict):
                raise ValueError("Expecting symmetry to be None (unset), True, or dict.")

        # Validate indices in symmetry dict
        for i, j in symmetry.items():
            if len(i) != len(j):
                raise ValueError("Non-matching length of symmetry index tuples.")
            for k in range(len(i)):
                if not (i[k] >= 0 and j[k] >= 0 and i[k] < shape[k] and j[k] < shape[k]):
                    raise ValueError("Symmetry dimensions out of bounds.")

        # Compute all index combinations for given shape
        indices = compute_indices(shape)

        # Compute mapping from indices to sub element number,
        # accounting for symmetry
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
        flattened_sub_element_mapping = [sub_element_mapping[index] for i,
                                         index in enumerate(indices)]

        # Compute value shape
        value_shape = shape

        # Compute reference value shape based on symmetries
        if symmetry:
            reference_value_shape = (product(shape) - len(symmetry),)
            self._mapping = "symmetries"
        else:
            reference_value_shape = shape
            self._mapping = sub_element.mapping()

        value_shape = value_shape + sub_element.value_shape
        reference_value_shape = reference_value_shape + sub_element.reference_value_shape
        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape,
                              reference_value_shape=reference_value_shape)
        self._family = sub_element.family()
        self._degree = sub_element.degree()
        self._sub_element = sub_element
        self._shape = shape
        self._symmetry = symmetry
        self._sub_element_mapping = sub_element_mapping
        self._flattened_sub_element_mapping = flattened_sub_element_mapping

        if variant is None:
            var_str = ""
        else:
            var_str = ", variant='" + variant + "'"

        # Cache repr string
        self._repr = (f"TensorElement({repr(sub_element)}, shape={shape}, "
                      f"symmetry={symmetry}{var_str})")

    @property
    def pull_back(self):
        """Get pull back."""
        if len(self._symmetry) > 0:
            symmetry = {}
            n = 0
            for i, j in self._symmetry.items():
                if j in symmetry:
                    symmetry[i] = symmetry[j]
                else:
                    symmetry[i] = n
                    symmetry[j] = n
                    n += 1
            for i in np.ndindex(self.value_shape):
                if i not in symmetry:
                    symmetry[i] = n
                    n += 1
            return SymmetricPullBack(self, symmetry)
        return super().pull_back

    def __repr__(self):
        """Doc."""
        return self._repr

    def variant(self):
        """Return the variant used to initialise the element."""
        return self._sub_element.variant()

    def mapping(self):
        """Doc."""
        return self._mapping

    def flattened_sub_element_mapping(self):
        """Doc."""
        return self._flattened_sub_element_mapping

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative.

        component index for a given component index.
        """
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)

        i = self.symmetry().get(i, i)
        l = len(self._shape)  # noqa: E741
        ii = i[:l]
        jj = i[l:]
        if ii not in self._sub_element_mapping:
            raise ValueError(f"Illegal component index {i}.")
        k = self._sub_element_mapping[ii]
        return (k, jj)

    def symmetry(self):
        r"""Return the symmetry dict, which is a mapping :math:`c_0 \\to c_1`.

        meaning that component :math:`c_0` is represented by component
        :math:`c_1`.
        A component is a tuple of one or more ints.
        """
        return self._symmetry

    def reconstruct(self, **kwargs):
        """Doc."""
        sub_element = self._sub_element.reconstruct(**kwargs)
        return TensorElement(sub_element, shape=self._shape, symmetry=self._symmetry)

    def __str__(self):
        """Format as string for pretty printing."""
        if self._symmetry:
            tmp = ", ".join("%s -> %s" % (a, b) for (a, b) in self._symmetry.items())
            sym = " with symmetries (%s)" % tmp
        else:
            sym = ""
        return ("<tensor element with shape %s of %s%s>" %
                (self.value_shape, self._sub_element, sym))

    def shortstr(self):
        """Format as string for pretty printing."""
        if self._symmetry:
            tmp = ", ".join("%s -> %s" % (a, b) for (a, b) in self._symmetry.items())
            sym = " with symmetries (%s)" % tmp
        else:
            sym = ""
        return "Tensor<%s x %s%s>" % (self.value_shape,
                                      self._sub_element.shortstr(), sym)
