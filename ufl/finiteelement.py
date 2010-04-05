"This module defines the UFL finite element classes."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03"

# Modified by Kristian B. Oelgaard
# Modified by Marie E. Rognes (meg@simula.no) 2010

# Last changed: 2010-03-22

from ufl.assertions import ufl_assert
from ufl.permutation import compute_indices
from ufl.elementlist import ufl_elements
from ufl.common import product, index_to_component, component_to_index, istr
from ufl.geometry import as_cell, domain2facet
from ufl.log import info_blue
from ufl.log import BLUE

class FiniteElementBase(object):
    "Base class for all finite elements"
    __slots__ = ("_family", "_cell", "_degree", "_value_shape", "_repr", "_domain")

    def __init__(self, family, cell, degree, value_shape):
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

    def family(self):
        "Return finite element family"
        return self._family

    def cell(self):
        "Return cell of finite element"
        return self._cell

    def set_cell(self, cell):
        "Set cell for element"
        self._cell = cell

    def degree(self):
        "Return polynomial degree of finite element"
        return self._degree

    def set_degree(self, degree):
        "Set degree for element"
        self._degree = degree

    def value_shape(self):
        "Return the shape of the value space"
        return self._value_shape

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
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
        r = len(self.value_shape())
        ufl_assert(len(i) == r,
                   "Illegal component index '%r' (value rank %d) for element (value rank %d)." % (i, len(i), r))

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __add__(self, other):
        "Add two elements, creating an enriched element"
        ufl_assert(isinstance(other, FiniteElementBase), "Can't add element and %s." % other.__class__)
        print
        print BLUE % "WARNING: Creating an EnrichedElement,\n         if you intended to create a MixedElement use '*' instead of '+'."
        print
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
        if isinstance(index, Measure) or isinstance(as_cell(index), Cell):
            return ElementRestriction(self, index)
        #if isinstance(index, int):
        #    return SubElement(self, index)
        return NotImplemented

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"
    def __init__(self, family, cell, degree):
        "Create finite element"
        cell = as_cell(cell)
        domain = cell.domain()

        # Check that the element family exists
        ufl_assert(family in ufl_elements, 'Unknown finite element "%s".' % family)

        # Check that element data is valid (and also get common family name)
        (family, self._short_name, value_rank, (kmin, kmax), domains) = ufl_elements[family]
        ufl_assert(domain in domains or domain is None,
                   'Domain "%s" invalid for "%s" finite element.' % (domain, family))
        ufl_assert(kmin is None or degree >= kmin or degree is None,
                   'Degree "%s" invalid for "%s" finite element.' % (istr(degree), family))
        ufl_assert(kmax is None or degree <= kmax or degree is None,
                   'Degree "%s" invalid for "%s" finite element.' % (istr(degree), family))

        # Set value dimension (default to using domain dimension in each axis)
        dim = cell.geometric_dimension()
        value_shape = (dim,)*(value_rank)

        # Initialize element data
        FiniteElementBase.__init__(self, family, cell, degree, value_shape)

        # Cache repr string
        self._repr = "FiniteElement(%r, %r, %s)" % (self.family(), self.cell(), istr(self.degree()))

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s%s on a %s>" % (self._short_name, istr(self.degree()), self.cell())

    def shortstr(self):
        "Format as string for pretty printing."
        return "%s%s" % (self._short_name, istr(self.degree()))

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
        ufl_assert(all(e.cell() == cell for e in elements), "Cell mismatch for sub elements of mixed element.")

        # Compute value shape
        value_size_sum= sum(product(s.value_shape()) for s in self._sub_elements)
        if "value_shape" in kwargs:
            value_shape = kwargs["value_shape"]
            # Validate value_shape
            if type(self) is MixedElement:
                ufl_assert(product(value_shape) == value_size_sum,
                    "Provided value_shape doesn't match the total "\
                    "value size of all subelements.")
        else:
            # Default value dimension: Treated simply as all subelement values unpacked in a vector.
            value_shape = (value_size_sum,)

        # Initialize element data
        degree = max(e.degree() for e in self._sub_elements)
        FiniteElementBase.__init__(self, "Mixed", cell, degree, value_shape)

        # Cache repr string
        self._repr = "MixedElement(*%r, **{'value_shape': %r })" % (self._sub_elements, self._value_shape)

    def set_cell(self, cell):
        "Set cell for element"
        self._cell = cell
        for element in self._sub_elements:
            element.set_cell(cell)

    def set_degree(self, degree):
        "Set degree for element"
        self._degree = degree
        for element in self._sub_elements:
            element.set_degree(degree)

    def num_sub_elements(self):
        "Return number of sub elements"
        return len(self._sub_elements)

    def sub_elements(self):
        "Return list of sub elements"
        return self._sub_elements

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        ufl_assert(len(i) > 0, "Illegal component index (empty).")

        # Indexing into a long vector
        if len(self.value_shape()) == 1:
            j, = i
            ufl_assert(j < product(self.value_shape()), "Illegal component index (value %d)." % j)
            # Find subelement for this index
            for e in self._sub_elements:
                sh = e.value_shape()
                si = product(sh)
                if j < si:
                    break
                j -= si
            ufl_assert(j >= 0, "Moved past last value component!")
            # Convert index into a shape tuple
            i = index_to_component(j, sh)
            return e.extract_component(i)

        # Indexing into a multidimensional tensor
        ufl_assert(i[0] < len(self._sub_elements), "Illegal component index (dimension %d)." % i[0])
        return self._sub_elements[i[0]].extract_component(i[1:])

    def __str__(self):
        "Format as string for pretty printing."
        return "<Mixed element: (" + ", ".join(str(element) for element in self._sub_elements) + ")" + ">"

    def shortstr(self):
        "Format as string for pretty printing."
        return "Mixed<" + ", ".join(element.shortstr() for element in self._sub_elements) + ">"

class VectorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"

    def __init__(self, family, cell, degree, dim=None):
        "Create vector element (repeated mixed element)"

        cell = as_cell(cell)

        # Set default size if not specified
        if dim is None:
            dim = cell.geometric_dimension()

        # Create mixed element from list of finite elements
        sub_element = FiniteElement(family, cell, degree)
        sub_elements = [sub_element]*dim

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Compute value shape
        value_shape = (dim,) + sub_element.value_shape()

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = degree
        self._sub_element = sub_element

        self._repr = "VectorElement(%r, %r, %s, %d)" % \
            (self._family, self._cell, str(self._degree), len(self._sub_elements))

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

    def __init__(self, family, cell, degree, shape=None, symmetry=None):
        "Create tensor element (repeated mixed element with optional symmetries)"
        cell = as_cell(cell)

        # Set default shape if not specified
        if shape is None:
            dim = cell.geometric_dimension()
            shape = (dim, dim)

            # Construct default symmetry for matrix elements
            if symmetry == True:
                symmetry = dict( ((i,j), (j,i)) for i in range(dim) for j in range(dim) if i > j )
            else:
                ufl_assert(symmetry in (None, True),
                          "Symmetry of tensor element cannot be specified unless shape has been specified.")

        # Compute all index combinations for given shape
        indices = compute_indices(shape)

        # Compute sub elements and mapping from indices to sub elements, accounting for symmetry
        sub_element = FiniteElement(family, cell, degree)
        sub_elements = []
        sub_element_mapping = {}
        for index in indices:
            if symmetry and index in symmetry:
                continue
            sub_element_mapping[index] = len(sub_elements)
            sub_elements += [sub_element]

        # Get common family name (checked in FiniteElement.__init__)
        family = sub_element.family()

        # Update mapping for symmetry
        for index in indices:
            if symmetry and index in symmetry:
                sub_element_mapping[index] = sub_element_mapping[symmetry[index]]

        # Compute value shape
        value_shape = shape + sub_element.value_shape()

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = degree
        self._sub_element = sub_element
        self._shape = shape
        self._symmetry = symmetry
        self._sub_element_mapping = sub_element_mapping

        # Cache repr string
        self._repr = "TensorElement(%r, %r, %r, %r, %r)" % \
            (self._family, self._cell, self._degree, self._shape, self._symmetry)

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        l = len(self._shape)
        ii = i[:l]
        jj = i[l:]
        ufl_assert(ii in self._sub_element_mapping, "Illegal component index %s." % repr(i))
        subelement = self._sub_elements[self._sub_element_mapping[ii]]
        return subelement.extract_component(jj)

    def symmetry(self):
        """Return the symmetry dict, which is a mapping c0 -> c1
        meaning that component c0 is represented by component c1."""
        return self._symmetry

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

        value_shape = elements[0].value_shape()
        ufl_assert(all(e.value_shape() == value_shape for e in elements), "Element value shape mismatch.")

        # Initialize element data
        FiniteElementBase.__init__(self, "EnrichedElement", cell, degree, value_shape)

        # Cache repr string
        self._repr = "EnrichedElement(%s)" % ", ".join(repr(e) for e in self._elements)

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(str(e) for e in self._elements)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>" % " + ".join(e.shortstr() for e in self._elements)

class ElementRestriction(FiniteElementBase):
    def __init__(self, element, domain):
        ufl_assert(isinstance(element, FiniteElementBase), "Expecting a finite element instance.")
        from ufl.integral import Measure
        from ufl.geometry import Cell
        ufl_assert(isinstance(domain, Measure) or isinstance(as_cell(domain), Cell),\
            "Expecting a subdomain represented by a Measure or Cell instance.")
        FiniteElementBase.__init__(self, "ElementRestriction", element.cell(), element.degree(), element.value_shape())
        self._element = element

        # Just attach domain if it is a Measure
        if isinstance(domain, Measure):
            self._domain = domain
        else:
            # Create Cell (if we get a string)
            domain = as_cell(domain)
            # Check for facet and handle it
            if domain.domain() == "facet":
                domain = Cell(domain2facet[self.cell().domain()])
            self._domain = domain

        self._repr = "ElementRestriction(%r, %r)" % (element, domain)

    def element(self):
        "Return the element which is restricted."
        return self._element

    def __str__(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element, self._domain)

    def shortstr(self):
        "Format as string for pretty printing."
        return "<%s>|_{%s}" % (self._element.shortstr(), self._domain)
