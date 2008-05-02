"This module defines the UFL finite element classes."

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03 -- 2008-05-02"

from output import ufl_assert
from elements import ufl_elements

# Map from valid domains to their topological dimension
_domain2dim = {"interval": 1, "triangle": 2, "tetrahedron": 3, "quadrilateral": 2, "hexahedron": 3}

class FiniteElementBase(object):
    "Base class for all finite elements"
    def __add__(self, other):
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

        # Save element data
        self.family = family
        self.domain = domain
        self.degree = degree
        self._value_rank = value_rank

    def value_rank(self):
        return self._value_rank

    def __repr__(self):
        return "FiniteElement(%s, %s, %d)" % (repr(self.family), repr(self.domain), self.degree)

class MixedElement(FiniteElementBase):
    "A finite element composed of a nested hierarchy of mixed or simple elements"

    def __init__(self, *elements):
        "Create mixed finite element from given list of elements"
        ufl_assert(all(e.domain == elements[0].domain for e in elements), "Domain mismatch for mixed element.")
        self._elements = elements

    def __repr__(self):
        return "MixedElement(*%s)" % repr(self._elements)

class VectorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"

    def __init__(self, family, domain, degree, size=None):
        "Create vector element (repeated mixed element)"

        if size is None:
            size = _domain2dim[domain]

        # Create mixed element from list of finite elements
        subelement = FiniteElement(family, domain, degree)
        MixedElement.__init__(self, *[subelement for i in range(size)])

        # Save data
        self.family = family
        self.domain = domain
        self.degree = degree
        self.size   = size

        self._value_rank = subelement.value_rank() + 1

    def value_rank(self):
        return self._value_rank

    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.domain), self.degree, repr(self.size))

class TensorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    
    def __init__(self, family, domain, degree, shape=None, symmetric=False):
        "Create tensor element (repeated mixed element)"

        def product(l):
            import operator
            return reduce(operator.__mul__, l)

        if shape is None:
            dim = _domain2dim[domain]
            shape = (dim, dim)

        # Create nested mixed element recursively
        subelement = FiniteElement(family, domain, degree)
        MixedElement.__init__(self, *[subelement for i in range(product(shape))])
        
        # Save data
        self.family    = family
        self.domain    = domain
        self.degree    = degree
        self.shape     = shape
        self.symmetric = symmetric

        if subelement.value_rank() != 0:
            ufl_warning("Creating a tensor element of nonscalar elements, this is not tested (if it even makes sense).")

        self._value_rank = subelement.value_rank() + len(shape)

    def value_rank(self):
        return self._value_rank

    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s, %s)" % (repr(self.family), repr(self.domain), self.degree, repr(self.shape), repr(self.symmetric))
