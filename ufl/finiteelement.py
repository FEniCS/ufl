"This module defines the UFL finite element classes."

# Copyright (C) 2008-2012 Martin Sandve Alnes
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
# Last changed: 2012-08-16

from itertools import izip
from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.elementlist import ufl_elements, aliases
from ufl.common import product, index_to_component, component_to_index, istr, EmptyDict
from ufl.geometry import as_cell, domain2facet, ProductCell
from ufl.log import info_blue, warning, warning_blue, error

class FiniteElementBase(object):
    "Base class for all finite elements"
    __slots__ = ("_family", "_cell", "_degree", "_quad_scheme", "_value_shape",
                 "_repr", "_domain")

    def __init__(self, family, cell, degree, quad_scheme, value_shape):
        "Initialize basic finite element data"
        ufl_assert(isinstance(family, str), "Invalid family type.")
        cell = as_cell(cell)
        ufl_assert(isinstance(degree, int) or degree is None, "Invalid degree type.")
        ufl_assert(isinstance(value_shape, tuple), "Invalid value_shape type.")
        self._family = family
        self._cell = cell
        self._degree = degree
        self._value_shape = value_shape
        self._domain = None
        self._quad_scheme = quad_scheme

    def family(self):
        "Return finite element family"
        return self._family

    def cell(self):
        "Return cell of finite element"
        return self._cell

    def is_cellwise_constant(self):
        "Return whether the basis functions of this element is spatially constant over each cell."
        return self.family() == "Real" or self.degree() == 0

    def degree(self):
        "Return polynomial degree of finite element"
        return self._degree

    def quadrature_scheme(self):
        "Return quadrature scheme of finite element"
        return self._quad_scheme

    def value_shape(self):
        "Return the shape of the value space"
        return self._value_shape

    def symmetry(self):
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return EmptyDict

    def extract_subelement_component(self, i):
        """Extract direct subelement index and subelement relative
        component index for a given component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (None, i)

    def extract_component(self, i):
        """Recursively extract component index relative to a (simple) element
        and that element for given value component index"""
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (i, self)

    def domain_restriction(self):
        "Return the domain onto which the element is restricted."
        return self._domain

    def num_sub_elements(self):
        "Return number of sub elements"
        return 0

    def sub_elements(self):
        "Return list of sub elements"
        return []

    def _check_component(self, i):
        "Check that component index i is valid"
        sh = self.value_shape()
        r = len(sh)
        if not (len(i) == r and all(j < k for (j,k) in izip(i, sh))):
            error(("Illegal component index '%r' (value rank %d)" + \
                   "for element (value rank %d).") % (i, len(i), r))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __add__(self, other):
        "Add two elements, creating an enriched element"
        ufl_assert(isinstance(other, FiniteElementBase), "Can't add element and %s." % other.__class__)
        warning_blue("WARNING: Creating an EnrichedElement,\n         " +\
                     "if you intended to create a MixedElement use '*' instead of '+'.")
        return EnrichedElement(self, other)

    def __repr__(self):
        "Format as string for evaluation as Python object."
        return self._repr

    def __mul__(self, other):
        "Multiply two elements, creating a mixed element"
        ufl_assert(isinstance(other, FiniteElementBase), "Can't multiply element and %s." % other.__class__)
        return MixedElement(self, other)

    def __getitem__(self, index):
        "Restrict finite element to a subdomain, subcomponent or topology (cell)."
        from ufl.integral import Measure
        from ufl.geometry import Cell
        if isinstance(index, (Measure, Cell)) or\
                index == "facet" or\
                isinstance(as_cell(index), Cell): # TODO: Can we just drop the as_cell call?
            return RestrictedElement(self, index)
        #if isinstance(index, int):
        #    return SubElement(self, index)
        return NotImplemented

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"
    def __init__(self, family, cell=None, degree=None, quad_scheme=None,
                 form_degree=None):
        """Create finite element

        *Arguments*
            family (string)
               The finite element family
            cell
               The cell
            degree (int)
               The polynomial degree (optional)
            quad_scheme
               The quadrature scheme (optional)
            form_degree (int)
               The form degree (FEEC notation, used when field is
               viewed as k-form)
        """

        # Map evt. string argument to a Cell
        cell = as_cell(cell)

        # Check whether this family is an alias for something else
        if family in aliases:
            (name, cell, r) = aliases[family](family, cell, degree, form_degree)
            #info_blue("%s, is an alias for %s " % (
            #        (family, cell, degree, form_degree),
            #        (name, cell, r)))
            self.__init__(name, cell, r, quad_scheme)
            return

        # Check that the element family exists
        ufl_assert(family in ufl_elements,
                   'Unknown finite element "%s".' % family)

        # Check that element data is valid (and also get common family name)
        (family, self._short_name, value_rank, krange, domains) =\
            ufl_elements[family]

        # Validate domain if a valid cell is specified
        if cell.is_undefined():
            # Case of invalid cell, some stuff is then undefined,
            # such as the domain and some dimensions
            pass
        else:
            domain = cell.domain()
            ufl_assert(domain in domains,
                       'Domain "%s" invalid for "%s" finite element.' % (domain, family))

        # Validate degree if specified
        if degree is not None:
            ufl_assert(krange is not None,
                       'Degree "%s" invalid for "%s" finite element, '\
                           'should be None.' % (degree, family))
            kmin, kmax = krange
            ufl_assert(kmin is None or degree >= kmin,
                       'Degree "%s" invalid for "%s" finite element.' %\
                           (degree, family))
            ufl_assert(kmax is None or degree <= kmax,
                   'Degree "%s" invalid for "%s" finite element.' %\
                           (istr(degree), family))

        # Set value dimension (default to using domain dimension in each axis)
        if value_rank == 0:
            value_shape = ()
        else:
            ufl_assert(not cell.is_undefined(),
                       "Cannot infer value shape with an undefined cell.")
            dim = cell.geometric_dimension()
            value_shape = (dim,)*value_rank

        # Initialize element data
        super(FiniteElement, self).__init__(family, cell, degree,
                                            quad_scheme, value_shape)

        # Cache repr string
        self._repr = "FiniteElement(%r, %r, %r, %r)" % (
            self.family(), self.cell(), self.degree(), self.quadrature_scheme())

    def reconstruct(self, **kwargs):
        """Construct a new FiniteElement object with some properties
        replaced with new values."""
        kwargs["family"] = kwargs.get("family", self.family())
        kwargs["cell"] = kwargs.get("cell", self.cell())
        kwargs["degree"] = kwargs.get("degree", self.degree())
        kwargs["quad_scheme"] = kwargs.get("quad_scheme", self.quadrature_scheme())
        return FiniteElement(**kwargs)

    def __str__(self):
        "Format as string for pretty printing."
        qs = self.quadrature_scheme()
        qs = "" if qs is None else "(%s)" % qs
        return "<%s%s%s on a %s>" % (self._short_name, istr(self.degree()),\
                                           qs, self.cell())

    def shortstr(self):
        "Format as string for pretty printing."
        return "%s%s(%s)" % (self._short_name, istr(self.degree()), istr(self.quadrature_scheme()))

class MixedElement(FiniteElementBase):
    "A finite element composed of a nested hierarchy of mixed or simple elements"
    __slots__ = ("_sub_elements",)

    def __init__(self, *elements, **kwargs):
        "Create mixed finite element from given list of elements"

        # Unnest arguments if we get a single argument with a list of elements
        if len(elements) == 1 and isinstance(elements[0], (tuple, list)):
            elements = elements[0]
        self._sub_elements = list(elements)

        # Check that all elements are defined on the same domain
        cell = elements[0].cell()
        ufl_assert(all(e.cell() == cell for e in elements),
                   "Cell mismatch for sub elements of mixed element.")

        # Check that all elements use the same quadrature scheme
        # TODO: We can allow the scheme not to be defined.
        quad_scheme = elements[0].quadrature_scheme()
        ufl_assert(all(e.quadrature_scheme() == quad_scheme for e in elements),\
            "Quadrature scheme mismatch for sub elements of mixed element.")

        # Compute value shape
        value_size_sum = sum(product(s.value_shape()) for s in self._sub_elements)
        # Default value dimension: Treated simply as all subelement values unpacked in a vector.
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
        super(MixedElement, self).__init__("Mixed", cell, degree,
                                           quad_scheme, value_shape)

        # Cache repr string
        self._repr = "MixedElement(*%r, **{'value_shape': %r })" %\
            (self._sub_elements, self._value_shape)

    def is_cellwise_constant(self):
        "Return whether the basis functions of this element is spatially constant over each cell."
        return all(e.is_cellwise_constant() for e in self.sub_elements())

    def reconstruct(self, **kwargs):
        """Construct a new MixedElement object with some properties
        replaced with new values."""
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
        ufl_assert(j == product(self.value_shape()), "Size mismatch in symmetry algorithm.")
        return sm or EmptyDict

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

    def __str__(self):
        "Format as string for pretty printing."
        return "<Mixed element: (" + ", ".join(str(element) for element in self._sub_elements) + ")" + ">"

    def shortstr(self):
        "Format as string for pretty printing."
        return "Mixed<" + ", ".join(element.shortstr() for element in self._sub_elements) + ">"

class VectorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"

    def __init__(self, family, cell, degree, dim=None, quad_scheme=None,
                 form_degree=None):
        """
        Create vector element (repeated mixed element)

        *Arguments*
            family (string)
               The finite element family
            cell
               The cell
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

        cell = as_cell(cell)

        # Set default size if not specified
        if dim is None:
            ufl_assert(not cell.is_undefined(), "Cannot infer dimension with an undefined cell.")
            dim = cell.geometric_dimension()

        # Create mixed element from list of finite elements
        sub_element = FiniteElement(family, cell, degree, quad_scheme,
                                    form_degree)
        sub_elements = [sub_element]*dim

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Compute value shape
        #shape = () if dim == 1 else (dim,) # FIXME: Should we do this?
        shape = (dim,)
        value_shape = shape + sub_element.value_shape()

        # Initialize element data
        super(VectorElement, self).__init__(sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = degree
        self._sub_element = sub_element

        # Cache repr string
        self._repr = "VectorElement(%r, %r, %r, %d, %r)" % \
            (self._family, self._cell, self._degree, len(self._sub_elements), quad_scheme)

    def reconstruct(self, **kwargs):
        kwargs["family"] = kwargs.get("family", self.family())
        kwargs["cell"] = kwargs.get("cell", self.cell())
        kwargs["degree"] = kwargs.get("degree", self.degree())
        ufl_assert("dim" not in kwargs, "Cannot change dim in reconstruct.")
        kwargs["dim"] = len(self._sub_elements)
        kwargs["quad_scheme"] = kwargs.get("quad_scheme", self.quadrature_scheme())
        return VectorElement(**kwargs)

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s vector element of degree %s on a %s: %d x %s>" % \
               (self.family(), istr(self.degree()), self.cell(), len(self._sub_elements), self._sub_element)

    def shortstr(self):
        "Format as string for pretty printing."
        return "Vector<%d x %s>" % (len(self._sub_elements), self._sub_element.shortstr())

class TensorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    #__slots__ = ("_family", "_cell", "_degree", "_value_shape")
    __slots__ = ("_sub_element", "_shape", "_symmetry", "_sub_element_mapping",)

    def __init__(self, family, cell, degree, shape=None, symmetry=None, quad_scheme=None):
        "Create tensor element (repeated mixed element with optional symmetries)"
        cell = as_cell(cell)

        # Set default shape if not specified
        if shape is None:
            ufl_assert(not cell.is_undefined(), "Cannot infer value shape with an undefined cell.")
            dim = cell.geometric_dimension()
            #shape = () if dim == 1 else (dim, dim) # FIXME: Should we do this?
            shape = (dim, dim)

        # Construct default symmetry for matrix elements
        if symmetry == True:
            symmetry = dict( ((i,j), (j,i)) for i in range(dim) for j in range(dim) if i > j )

        # Validate indices in symmetry dict
        if isinstance(symmetry, dict):
            for i,j in symmetry.iteritems():
                ufl_assert(len(i) == len(j), "Non-matching length of symmetry index tuples.")
                for k in range(len(i)):
                    ufl_assert(i[k] >= 0 and j[k] >= 0 and i[k] < shape[k] and j[k] < shape[k],
                               "Symmetry dimensions out of bounds.")
        else:
            ufl_assert(symmetry is None, "Expecting symmetry to be None, True, or dict.")

        # Compute all index combinations for given shape
        indices = compute_indices(shape)

        # Compute sub elements and mapping from indices to sub elements, accounting for symmetry
        sub_element = FiniteElement(family, cell, degree, quad_scheme)
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
            (self._family, self._cell, self._degree, self._shape, self._symmetry, quad_scheme)

    def reconstruct(self, **kwargs):
        kwargs["family"] = kwargs.get("family", self.family())
        kwargs["cell"]   = kwargs.get("cell",   self.cell())
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
        ufl_assert(ii in self._sub_element_mapping, "Illegal component index %s." % repr(i))
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
            sym = " with symmetries (%s)" % ", ".join("%s -> %s" % (a,b) for (a,b) in self._symmetry.iteritems())
        elif self._symmetry:
            sym = " with symmetry"
        return "<%s tensor element of degree %s and shape %s on a %s%s>" % \
            (self.family(), istr(self.degree()), self.value_shape(), self.cell(), sym)

    def shortstr(self):
        "Format as string for pretty printing."
        sym = ""
        if isinstance(self._symmetry, dict):
            sym = " with symmetries (%s)" % ", ".join("%s -> %s" % (a,b) for (a,b) in self._symmetry.iteritems())
        elif self._symmetry:
            sym = " with symmetry"
        return "Tensor<%s x %s%s>" % (self.value_shape(), self._sub_element.shortstr(), sym)

class EnrichedElement(FiniteElementBase):
    """The vector sum of two finite element spaces:

        EnrichedElement(V, Q) = {v + q | v in V, q in Q}.
    """
    def __init__(self, *elements):
        self._elements = elements

        cell = elements[0].cell()
        ufl_assert(all(e.cell() == cell for e in elements), "Element cell mismatch.")

        degree = max(e.degree() for e in elements)

        # We can allow the scheme not to be defined, but all defined should be equal
        quad_schemes = [e.quadrature_scheme() for e in elements]
        quad_schemes = [qs for qs in quad_schemes if qs is not None]
        quad_scheme = quad_schemes[0] if quad_schemes else None
        ufl_assert(all(qs == quad_scheme for qs in quad_schemes),\
            "Quadrature scheme mismatch.")

        value_shape = elements[0].value_shape()
        ufl_assert(all(e.value_shape() == value_shape for e in elements), "Element value shape mismatch.")

        # Initialize element data
        super(EnrichedElement, self).__init__("EnrichedElement", cell, degree, quad_scheme, value_shape)

        # Cache repr string
        self._repr = "EnrichedElement(%s)" % ", ".join(repr(e) for e in self._elements)

    def reconstruct(self, **kwargs):
        """Construct a new EnrichedElement object with some properties
        replaced with new values."""
        elements = [e.reconstruct(**kwargs) for e in self._elements]
        if all(a == b for (a,b) in izip(elements, self._elements)):
            return self
        return EnrichedElement(*elements)

    def is_cellwise_constant(self):
        "Return whether the basis functions of this element is spatially constant over each cell."
        return all(e.is_cellwise_constant() for e in self._elements)

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(str(e) for e in self._elements)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(e.shortstr() for e in self._elements)

class RestrictedElement(FiniteElementBase):
    def __init__(self, element, domain):
        ufl_assert(isinstance(element, FiniteElementBase), "Expecting a finite element instance.")
        from ufl.integral import Measure
        from ufl.geometry import Cell
        ufl_assert(isinstance(domain, Measure) or domain == "facet"\
                   or isinstance(as_cell(domain), Cell),\
            "Expecting a subdomain represented by a Measure, a Cell instance, or the string 'facet'.")
        super(RestrictedElement, self).__init__("RestrictedElement", element.cell(),\
            element.degree(), element.quadrature_scheme(), element.value_shape())
        self._element = element

        # Just attach domain if it is a Measure or Cell
        if isinstance(domain, (Measure, Cell)):
            self._domain = domain
        else:
            # Check for facet and handle it
            if domain == "facet":
                cell = self.cell()
                ufl_assert(not cell.is_undefined(), "Cannot determine facet cell of undefined cell.")
                domain = Cell(domain2facet[cell.domain()])
            else:
                # Create Cell (if we get a string)
                domain = as_cell(domain)
            self._domain = domain

        if isinstance(self._domain, Cell) and self._domain.is_undefined():
            warning("Undefined cell as domain in RestrictedElement. Not sure if this is well defined.")

        # Cache repr string
        self._repr = "RestrictedElement(%r, %r)" % (self._element, self._domain)

    def reconstruct(self, **kwargs):
        """Construct a new RestrictedElement object with some properties
        replaced with new values."""
        element = self._element.reconstruct(**kwargs)
        domain = kwargs.get("domain", self.domain())
        return RestrictedElement(element=element, domain=domain)

    def is_cellwise_constant(self):
        "Return whether the basis functions of this element is spatially constant over each cell."
        return self._element.is_cellwise_constant()

    def element(self):
        "Return the element which is restricted."
        return self._element

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element, self._domain)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element.shortstr(), self._domain)

    def symmetry(self):
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return self._element.symmetry()

    def num_sub_elements(self):
        "Return number of sub elements"
        return self._element.num_sub_elements()
        #return 1

    def sub_elements(self):
        "Return list of sub elements"
        return self._element.sub_elements()
        #return [self._element]


class TensorProductElement(FiniteElementBase):
    """The tensor product of d element spaces:

    V = V_0 \otimes V_1 \otimes ...  \otimes V_d

    Given bases {phi_i} for V_i for i = 1, ...., d,
    { phi_0 * phi_1 *.... * phi_d } forms a basis for V.
    """
    __slots__ = ("_sub_elements",)

    def __init__(self, *elements):
        "Create TensorProductElement from a given list of elements."

        self._sub_elements = list(elements)
        self._repr = "TensorProductElement(*%r)" % self._sub_elements
        family = "TensorProductElement"

        # Define cell as the product of each subcell
        cell = ProductCell(*[e.cell() for e in self._sub_elements])

        # Define polynomial degree as the maximal of each subelement
        degree = max(e.degree() for e in self._sub_elements)

        # No quadrature scheme defined
        quad_scheme = None

        # For now, check that all subelements have the same value
        # shape, and use this.
        value_shape = self._sub_elements[0].value_shape()

        super(TensorProductElement, self).__init__(family, cell, degree,
                                                   quad_scheme, value_shape)
    def sub_elements(self):
        "Return subelements (factors)."
        return self._sub_elements

    def num_sub_elements(self):
        "Return number of subelements."
        return len(self._sub_elements)

    def __str__(self):
        "Pretty-print."
        return "TensorProductElement(%r)" % [str(e) for e in self.sub_elements()]

    def shortstr(self):
        "Short pretty-print."
        return "TensorProductElement(%r)"% [e.shortstr()
                                            for e in self.sub_elements()]

    def __repr__(self):
        return self._repr

