"This module defines the UFL finite element classes."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03 -- 2008-05-02"

from output import ufl_assert
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
        "Return value rank of finite element"
        return self._value_rank

    def __add__(self, other):
        "Add two elements, creating a mixed element"
        ufl_assert(isinstance(other, FiniteElementBase), "Can't add element and %s" % other.__class__)
        return MixedElement(self, other)

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"

    def __init__(self, family, domain, degree):
        "Create finite element"

        # Check that element exists
        ufl_assert(family in ufl_elements, 'Unknown finite element "%s".' % family)

        # Check that element data is valid (and also get common family name)
        (family, short_name, value_rank, (kmin, kmax), domains) = ufl_elements[family]
        ufl_assert(domain in domains,          'Domain "%s" invalid for "%s" finite element.' % (domain, family))
        ufl_assert(not kmin or degree >= kmin, 'Degree "%d" invalid for "%s" finite element.' % (degree, family))
        ufl_assert(not kmax or degree <= kmax, 'Degree "%d" invalid for "%s" finite element.' % (degree, family))

        # Initialize element data
        FiniteElementBase.__init__(self, family, domain, degree, value_rank)

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
        ufl_assert(all(e.domain() == elements[0].domain() for e in elements), "Domain mismatch for mixed element.")
        self._elements = elements

    def __repr__(self):
        "Return string representation"
        return "MixedElement(*%s)" % repr(self._elements)

    def __str__(self):
        "Pretty printing"
        return "Mixed element: [" + ", ".join(str(element) for element in self._elements) + "]"

class VectorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"

    def __init__(self, family, domain, degree, dim=None):
        "Create vector element (repeated mixed element)"

        # Set default size if not specified
        if dim is None:
            dim = _domain2dim[domain]

        # Create mixed element from list of finite elements
        sub_element = FiniteElement(family, domain, degree)
        MixedElement.__init__(self, *[sub_element for i in range(dim)])

        # Initialize element data
        FiniteElementBase.__init__(self, family, domain, degree, sub_element.value_rank() + 1)
        self._dim = dim

    def dim(self):
        "Return dimension of vector-valued element"
        return self._dim

    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self._family), repr(self._domain), self._degree, repr(self._dim))

    def __str__(self):
        "Pretty printing"
        return "%s vector element of degree %d and dimension %d on a %s" % (self.family(), self.degree(), self.dim(), self.domain())

class TensorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    
    def __init__(self, family, domain, degree, shape=None, is_symmetric=False):
        "Create tensor element (repeated mixed element)"

        def product(l):
            import operator
            return reduce(operator.__mul__, l)

        # Set default shape if not specified
        if shape is None:
            dim = _domain2dim[domain]
            shape = (dim, dim)

        # Create nested mixed element recursively
        sub_element = FiniteElement(family, domain, degree)
        MixedElement.__init__(self, *[sub_element for i in range(product(shape))])

        print MixedElement(sub_element, sub_element)

        #for dim in shape:
        #    print dim
        #    sub_element = MixedElement(*[sub_element for i in range(dim)])
        #print shape
        
        if sub_element.value_rank() != 0:
            ufl_warning("Creating a tensor element of nonscalar elements, this is not tested (if it even makes sense).")

        # Initialize element data
        FiniteElementBase.__init__(self, family, domain, degree, sub_element.value_rank() + len(shape))
        self._shape = shape
        self._is_symmetric = is_symmetric

    def shape(self):
        "Return shape of tensor element"
        return self._shape

    def is_symmetric(self):
        "Return True iff tensor element is symmetric"
        return self._is_symmetric

    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s, %s)" % (repr(self._family), repr(self._domain), self._degree, repr(self._shape), repr(self._is_symmetric))

    def __str__(self):
        "Pretty printing"
        return "%s tensor element of degree %d and shape %s on a %s" % (self.family(), self.degree(), str(self.shape()), self.domain())
