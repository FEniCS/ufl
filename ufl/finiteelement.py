"This module defines the UFL finite element classes."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03 -- 2008-06-03"

from output import ufl_assert
from permutation import compute_indices
from elements import ufl_elements

# Map from valid domains to their topological dimension
_domain2dim = {"interval": 1, "triangle": 2, "tetrahedron": 3, "quadrilateral": 2, "hexahedron": 3}

class FiniteElementBase(object):
    "Base class for all finite elements"

    def __init__(self, family, domain, degree, value_rank):
        "Initialize basic finite element data"
        self._family = family
        self._domain = domain
        self._degree = degree
        self._value_rank = value_rank

    def family(self):
        "Return finite element family"
        return self._family

    def domain(self):
        "Return domain of finite element"
        return self._domain

    def degree(self):
        "Return polynomial degree of finite element"
        return self._degree

    def value_rank(self):
        "Return the rank of the value space"
        return self._value_rank

    def value_dimension(self, i):
        "Return the dimension of the value space for axis i"
        ufl_assert(not self._value_dimension is None, "Unspecified value shape.")
        self._check_component(i)
        return self._value_dimension[i]

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
        self._check_component(i)
        return (i, self)

    def _check_component(self, i):
        "Check that component index i is valid"
        r = self.value_rank()
        ufl_assert(len(i) == r,
                   "Illegal component index '%s' (value rank %d) for element (value rank %d)." % (repr(i), len(i), r))

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

        # Initialize element data
        FiniteElementBase.__init__(self, family, domain, degree, value_rank)

        # Set value dimension
        dim = _domain2dim[domain]
        self._value_dimension = tuple(dim for d in range(value_rank))

    def __repr__(self):
        "Return string representation"
        return "FiniteElement(%s, %s, %d)" % (repr(self.family()), repr(self.domain()), self.degree())

    def __str__(self):
        "Pretty printing"
        return "%s finite element of degree %d on a %s" % (self.family(), self.degree(), self.domain())

class MixedElement(FiniteElementBase):
    "A finite element composed of a nested hierarchy of mixed or simple elements"

    def __init__(self, *elements):
        "Create mixed finite element from given list of elements"

        # Unnest arguments if we get a single argument with a list of elements
        if len(elements) == 1 and (isinstance(elements[0], tuple) or isinstance(elements[0], list)):
            elements = elements[0]

        # Check that all elements are defined on the same domain
        domain = elements[0].domain()
        ufl_assert(all(e.domain() == domain for e in elements), "Domain mismatch for sub elements of mixed element.")

        # Initialize element data
        FiniteElementBase.__init__(self, "Mixed", domain, None, len(elements))
        self._sub_elements = list(elements)

        # Set value dimension # FIXME: Resolve this issue for a non-vector/tensor mixed element.
        
        # Solution #1: Simply leave this undefined for a general MixedElement?
        # The form compiled could check "if type(element) is MixedElement" before using element.value_dimension(i)
        # (not "isinstance(element, MixedElement)", which would include the subclasses VectorElement/TensorElement)
        self._value_dimension = None
        
        # Solution #2: Sum the dimensions of subelements. (This doesn't make sense!)
        #self._value_dimension = sum(e._value_dimension for e in self._sub_elements)
        
        # Solution #3: Look to UFC!
        # In UFC we do handle this somehow, since finite_element knows the value shape.
        # I think we define the value rank differently there than above, and value shape
        # should be consistent with the value rank definition.

    def sub_elements(self):
        "Return list of sub elements"
        return self._sub_elements

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
        self._check_component(i)
        ufl_assert(i[0] < len(self._sub_elements), "Illegal component index (dimension %d)." % i[0])
        return self._sub_elements[i[0]].extract_component(i[1:])

    def __repr__(self):
        "Return string representation"
        return "MixedElement(*%s)" % repr(self._sub_elements)

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

        # Initialize element data
        MixedElement.__init__(self, sub_elements)
        self._degree = degree
        self._value_rank = 1 + sub_element.value_rank()
        self._value_dimension = (dim,) + sub_element._value_dimension
        self._sub_element = sub_element

    def __repr__(self):
        "Return string representation"
        return "VectorElement(%s, %s, %d, %s)" % \
               (repr(self._family), repr(self._domain), self._degree, len(self._sub_elements))

    def __str__(self):
        "Pretty printing"
        return "%s vector element of degree %d on a %s: %d x [%s]" % \
               (self.family(), self.degree(), self.domain(), len(self._sub_elements), str(self._sub_element))

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
        
        # Initialize element data
        MixedElement.__init__(self, sub_elements)
        self._degree = degree
        self._value_rank = len(shape) + sub_element.value_rank()
        self._value_dimension = shape + sub_element._value_dimension
        self._sub_element = sub_element
        self._shape = shape
        self._symmetry = symmetry
        self._sub_element_mapping = sub_element_mapping

    def extract_component(self, i):
        "Extract base component index and (simple) element for given component index"
        self._check_component(i)
        l = len(self._shape)
        ii = i[:l]
        jj = i[l:]
        ufl_assert(ii in self._sub_element_mapping, "Illegal component index %s." % str(i))
        return self._sub_element_mapping[ii].extract_component(jj)

    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s, %s)" % (repr(self._family), repr(self._domain), self._degree, repr(self._shape), repr(self._symmetry))

    def __str__(self):
        "Pretty printing"
        print self.family()
        print self.degree()
        print self.shape()
        print self.domain()
        return "%s tensor element of degree %d and shape %s on a %s" % (self.family(), self.degree(), str(self.shape()), self.domain())
