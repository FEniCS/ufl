#!/usr/bin/env python

"""This module defines the UFL finite element classes.

UFL provides an extensive list of predefined finite element
families. Users (or more likely form compilers) may register
new elements by calling the function register_element."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03 -- 2008-03-13"

from output import *

import operator

def product(l):
    return reduce(operator.__mul__, l)

# Map from valid polygons to their spatial dimension
_domain2dim = { "interval":1, "triangle":2, "tetrahedron":3, "quadrilateral":2, "hexahedron":3 }

# List of valid elements
ufl_elements = {}

# Function for registering new elements
def register_element(family, short_name, value_rank, degrees, domains):
    "Register new finite element family"
    ufl_assert(family not in ufl_elements, 'Finite element \"%s\" has already been registered.' % family)
    ufl_elements[family]     = (family, short_name, value_rank, degrees, domains)
    ufl_elements[short_name] = (family, short_name, value_rank, degrees, domains)

# Register valid elements
register_element("Lagrange",                     "CG",     0, (1, None),    ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))
register_element("Discontinuous Lagrange",       "DG",     0, (0, None),    ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))
register_element("Crouzeix-Raviart",             "CR",     0, (1, 1),       ("triangle", "tetrahedron"))
register_element("Brezzi-Douglas-Marini",        "BDM",    1, (1, None),    ("triangle", "tetrahedron"))
register_element("Brezzi-Douglas-Fortin-Marini", "BDFM",   1, (1, None),    ("triangle", "tetrahedron"))
register_element("Raviart-Thomas",               "RT",     1, (0, None),    ("triangle", "tetrahedron"))
register_element("Nedelec 1st kind H(div)",      "N1div",  1, (0, None),    ("triangle", "tetrahedron"))
register_element("Nedelec 2nd kind H(div)",      "N2div",  1, (1, None),    ("triangle", "tetrahedron"))
register_element("Nedelec 1st kind H(curl)",     "N1curl", 1, (0, None),    ("triangle", "tetrahedron"))
register_element("Nedelec 2nd kind H(curl)",     "N2curl", 1, (1, None),    ("triangle", "tetrahedron"))
# FIXME: is this ok? Don't need QuadratureElement.
register_element("Quadrature",                   "Q",      0, (None, None), ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))
# FIXME: functions evaluated on the boundary can't use quadrature points from the inside of the cell,
#        therefore we need to have a separate space for functions evaluated in quadrature points on the boundary:
register_element("Boundary Quadrature",          "BQ",     0, (None, None), ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))

class FiniteElementBase:
    "Base class for all finite elements"
    pass

    def value_rank(self):
        ufl_error("Not implemented: value_rank().")
    
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

        # FIXME: Anders: Do we need to store this? (Martin: We don't _need_ to, but I don't see why not. It's infinitely better than depending on the structure of ufl_elements other places in the code.)
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
        self.elements = elements

    def value_rank(self):
        ufl_error("Not implemented for general MixedElement instances.")

    def __repr__(self):
        return "MixedElement(*%s)" % repr(self.elements)


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

        self._value_rank = subelement.value_rank() + 2

    def value_rank(self):
        return self._value_rank

    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s, %s)" % (repr(self.family), repr(self.domain), self.degree, repr(self.shape), repr(self.symmetric))

