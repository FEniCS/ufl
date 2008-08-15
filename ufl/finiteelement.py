"This module defines the UFL finite element classes."

from __future__ import absolute_import

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03 -- 2008-08-15"

from .output import ufl_assert
from .permutation import compute_indices
from .elements import ufl_elements
from .common import product

# Map from valid domains to their topological dimension
_domain2dim = {"interval": 1, "triangle": 2, "tetrahedron": 3, "quadrilateral": 2, "hexahedron": 3}

class FiniteElementBase(object):
    "Base class for all finite elements"

    def __init__(self, family, domain, degree, value_shape):
        "Initialize basic finite element data"
        self._family = family
        self._domain = domain
        self._degree = degree
        self._value_shape = value_shape

    def family(self):
        "Return finite element family"
        return self._family

    def domain(self):
        "Return domain of finite element"
        return self._domain

    def degree(self):
        "Return polynomial degree of finite element"
        return self._degree

    def value_shape(self):
        "Return the shape of the value space"
        return self._value_shape

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        return (i, self)

    def _check_component(self, i):
        "Check that component index i is valid"
        r = len(self.value_shape())
        ufl_assert(len(i) == r,
                   "Illegal component index '%r' (value rank %d) for element (value rank %d)." % (i, len(i), r))

    def __add__(self, other):
        "Add two elements, creating a mixed element"
        ufl_assert(isinstance(other, FiniteElementBase), "Can't add element and %s." % other.__class__)
        return MixedElement(self, other)

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"

    def __init__(self, family, domain, degree):
        "Create finite element"

        # Check that the element family exists
        ufl_assert(family in ufl_elements, 'Unknown finite element "%s".' % family)

        # Check that element data is valid (and also get common family name)
        (family, short_name, value_rank, (kmin, kmax), domains) = ufl_elements[family]
        ufl_assert(domain in domains,              'Domain "%s" invalid for "%s" finite element.' % (domain, family))
        ufl_assert(kmin is None or degree >= kmin, 'Degree "%d" invalid for "%s" finite element.' % (degree, family))
        ufl_assert(kmax is None or degree <= kmax, 'Degree "%d" invalid for "%s" finite element.' % (degree, family))
        
        # Set value dimension (default to using domain dimension in each axis)
        dim = _domain2dim[domain]
        value_shape = tuple(dim for d in range(value_rank))
        
        # Initialize element data
        FiniteElementBase.__init__(self, family, domain, degree, value_shape)

    def __repr__(self):
        "Return string representation"
        return "FiniteElement(%r, %r, %d)" % (self.family(), self.domain(), self.degree())

    def __str__(self):
        "Pretty printing"
        return "%s finite element of degree %d on a %s" % (self.family(), self.degree(), self.domain())

class MixedElement(FiniteElementBase):
    "A finite element composed of a nested hierarchy of mixed or simple elements"

    def __init__(self, *elements, **kwargs):
        "Create mixed finite element from given list of elements"

        # Unnest arguments if we get a single argument with a list of elements
        if len(elements) == 1 and isinstance(elements[0], (tuple, list)):
            elements = elements[0]
        self._sub_elements = list(elements)

        # Check that all elements are defined on the same domain
        domain = elements[0].domain()
        ufl_assert(all(e.domain() == domain for e in elements), "Domain mismatch for sub elements of mixed element.")
        
        # Compute value shape
        if "value_shape" in kwargs:
            value_shape = kwargs["value_shape"]
        else:
            # Default value dimension: Treated simply as all subelement values unpacked in a vector.
            value_sizes = (product(s.value_shape()) for s in self._sub_elements)
            value_shape = (sum(value_sizes),)

        # Initialize element data
        FiniteElementBase.__init__(self, "Mixed", domain, None, value_shape)

    def sub_elements(self):
        "Return list of sub elements"
        return self._sub_elements

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
        if isinstance(i, int):
            i = (i,)
        self._check_component(i)
        ufl_assert(len(i) > 0, "Illegal component index (empty).")
        ufl_assert(i[0] < len(self._sub_elements), "Illegal component index (dimension %d)." % i[0])
        return self._sub_elements[i[0]].extract_component(i[1:])

    def __repr__(self):
        "Return string representation"
        return "MixedElement(*%r, value_shape=%r)" % (self._sub_elements, self._value_shape)

    def __str__(self):
        "Pretty printing"
        return "Mixed element: [" + ", ".join(str(element) for element in self._sub_elements) + "]"

class VectorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"

    def __init__(self, family, domain, degree, dim=None):
        "Create vector element (repeated mixed element)"

        # Set default size if not specified
        if dim is None:
            dim = _domain2dim[domain]

        # Create mixed element from list of finite elements
        sub_element = FiniteElement(family, domain, degree)
        sub_elements = [sub_element]*dim

        # Compute value shape
        value_shape = (dim,) + sub_element.value_shape()

        # Initialize element data
        MixedElement.__init__(self, sub_elements, value_shape=value_shape)
        self._family = family
        self._degree = degree
        self._sub_element = sub_element

    def __repr__(self):
        "Return string representation"
        return "VectorElement(%r, %r, %d, %s)" % \
               (self._family, self._domain, self._degree, len(self._sub_elements))

    def __str__(self):
        "Pretty printing"
        return "%s vector element of degree %d on a %s: %d x [%s]" % \
               (self.family(), self.degree(), self.domain(), len(self._sub_elements), self._sub_element)

class TensorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    
    def __init__(self, family, domain, degree, shape=None, symmetry=None):
        "Create tensor element (repeated mixed element with optional symmetries)"
        
        # Set default shape if not specified
        if shape is None:
            dim = _domain2dim[domain]
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
        sub_element = FiniteElement(family, domain, degree)
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

    def __repr__(self):
        return "TensorElement(%r, %r, %r, %r, %r)" % (self._family, self._domain, self._degree, self._shape, self._symmetry)

    def __str__(self):
        "Pretty printing"
        print self.family()
        print self.degree()
        print self.shape()
        print self.domain()
        return "%s tensor element of degree %d and shape %s on a %s" % (self.family(), self.degree(), self.shape(), self.domain())
