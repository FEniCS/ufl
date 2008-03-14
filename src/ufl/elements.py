#!/usr/bin/env python

"""This module defines the UFL finite element classes.

UFL provides an extensive list of predefined finite element
families. Users (or more likely form compilers) may register
new elements by calling the function register_element."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03 -- 2008-03-13"

from io import *
from shapes import *

# List of valid elements
ufl_elements = {}

# Function for registering new elements
def register_element(family, short_name, value_rank, degrees, domains):
    "Register new finite element family"
    ufl_assert(family not in ufl_elements, 'Finite element \"%s\" has already been registered.' % family)
    ufl_elements[family] = (family, short_name, value_rank, degrees, domains)
    ufl_elements[short_name] = (family, short_name, value_rank, degrees, domains)

# Register valid elements
register_element("Lagrange",                     "CG",     0, (1, None), ("interval", "triangle", "tetrahedron"))
register_element("Discontinuous Lagrange",       "DG",     0, (0, None), ("interval", "triangle", "tetrahedron"))
register_element("Crouzeix-Raviart",             "CR",     0, (1, 1),    ("triangle", "tetrahedron"))
register_element("Brezzi-Douglas-Marini",        "BDM",    1, (1, None), ("triangle", "tetrahedron"))
register_element("Brezzi-Douglas-Fortin-Marini", "BDFM",   1, (1, None), ("triangle", "tetrahedron"))
register_element("Raviart-Thomas",               "RT",     1, (0, None), ("triangle", "tetrahedron"))
register_element("Nedelec 1st kind H(div)",      "N1div",  1, (0, None), ("triangle", "tetrahedron"))
register_element("Nedelec 2nd kind H(div)",      "N2div",  1, (1, None), ("triangle", "tetrahedron"))
register_element("Nedelec 1st kind H(curl)",     "N1curl", 1, (0, None), ("triangle", "tetrahedron"))
register_element("Nedelec 2nd kind H(curl)",     "N2curl", 1, (1, None), ("triangle", "tetrahedron"))

class FiniteElementBase:
    "Base class for all finite elements"
    pass

class FiniteElement(FiniteElementBase):
    "The basic finite element class for all simple finite elements"
    
    def __init__(self, family, domain, degree):
        "Create finite element"

        # Check that element exists
        ufl_assert(family in ufl_elements, 'Unknown finite element \"%s\".' % family)

        # Check that element data is valid (and also get common family name)
        (family, short_name, value_rank, (kmin, kmax), domains) = ufl_elements["family"]
        ufl_assert(domain in domains, 'Domain \"%s\" invalid for \"%s\" finite element.' % (domain, family))
        ufl_assert(not kmin or degree >= kmin, 'Degree \"%d\" invalid for \"%s\" finite element.' % (degree, family))
        ufl_assert(not kmax or degree <= kmax, 'Degree \"%d\" invalid for \"%s\" finite element.' % (degree, family))

        # Save element data
        self.family = family
        self.domain = domain
        self.degree = degree

        # FIXME: Do we need to store this?
        self.value_rank = value_rank
    
    def __repr__(self):
        return "FiniteElement(%s, %s, %d)" % (repr(self.family), repr(self.domain), self.degree)

class MixedElement(FiniteElementBase):
    "A finite element composed of a nested hierarchy of mixed or simple elements"
    
    def __init__(self, *elements):
        "Create mixed finite element from given list of elements"
        ufl_assert(all(e.domain == elements[0].domain for e in elements), "Domain mismatch for mixed element.")
        self.elements = elements

class VectorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    
    def __init__(self, family, domain, degree, size):
        "Create vector element (repeated mixed element)"

        # Create mixed element from list of finite elements
        MixedElement.__init__(self, [FiniteElement(family, domain, degree) for i in range(size)])

        # Save data
        self.family = family
        self.domain = domain
        self.degree = degree
        self.size = size
    
    def __repr__(self):
        return "VectorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.polygon), self.order, repr(self.size))

class TensorElement(MixedElement):
    "A special case of a mixed finite element where all elements are equal"
    
    def __init__(self, family, domain, degree, shape):
        "Create tensor element (repeated mixed element)"

        # Create nested mixed element recursively
        if len(shape) == 1:
            MixedElement.__init__(self, [FiniteElement(family, domain, degree) for i in range(shape[0])])
        else:
            MixedElement.__init__(self, [TensorElement(family, domain, degree, shape[1:]) for i in range(shape[0])])


        # Save data
        self.family = family
        self.domain = domain
        self.degree = degree
        self.shape = shape

    def __repr__(self):
        return "TensorElement(%s, %s, %d, %s)" % (repr(self.family), repr(self.domain), self.degree, repr(self.shape))

class QuadratureElement(FiniteElementBase):
    "A special finite element which can only be evaluated at quadrature points"

    def __init__(self, shape, domain_type="cell"):
        "Create quadrature element"
        FiniteElementBase.__init__(self, polygon)
        # FIXME: What is domain_type?
        self.domain_type = domain_type # TODO: define this better
