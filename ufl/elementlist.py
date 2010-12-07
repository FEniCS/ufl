"""This module provides an extensive list of predefined finite element
families. Users or more likely, form compilers, may register new
elements by calling the function register_element."""

__authors__ = "Martin Sandve Alnes and Anders Logg"
__date__ = "2008-03-03 -- 2009-11-16"

# Modified by Marie E. Rognes (meg@simula.no), 2010

from ufl.assertions import ufl_assert
from ufl.feec import FEEC_aliases

# List of valid elements
ufl_elements = {}

# Aliases: aliases[name] (...) -> (standard_name, ...)
aliases = {}

# Function for registering new elements
def register_element(family, short_name, value_rank, degree_range, domains):
    "Register new finite element family"
    ufl_assert(family not in ufl_elements, 'Finite element \"%s\" has already been registered.' % family)
    ufl_elements[family] = (family, short_name, value_rank, degree_range, domains)
    ufl_elements[short_name] = (family, short_name, value_rank, degree_range, domains)

def register_alias(alias, to):
    aliases[alias] = to

def show_elements():
    print "Showing all registered elements:"
    for k in sorted(ufl_elements.keys()):
        (family, short_name, value_rank, degree_range, domains) = ufl_elements[k]
        print
        print "Finite Element Family: %s, %s" % (repr(family), repr(short_name))
        print "Value rank:   ", value_rank
        print "Degree range: ", degree_range
        print "Defined on domains:" , domains

register_element("Lagrange", "CG", 0, (1, None),
                 ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))

register_element("Discontinuous Lagrange", "DG", 0, (0, None),
                 ("interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))

register_element("Bubble", "B", 0, (2, None),
                 ("interval", "triangle", "tetrahedron"))

register_element("Crouzeix-Raviart", "CR", 0, (1, 1),
                 ("triangle", "tetrahedron"))

register_element("Brezzi-Douglas-Marini", "BDM", 1, (1, None),
                 ("triangle", "tetrahedron"))

register_element("Brezzi-Douglas-Fortin-Marini", "BDFM", 1, (1, None),
                 ("triangle", "tetrahedron"))

register_element("Raviart-Thomas", "RT", 1, (1, None),
                 ("triangle", "tetrahedron"))

register_element("Morley", "MOR", 0, (None, None),
                 ("triangle", "tetrahedron"))

register_element("Nedelec 1st kind H(curl)", "N1curl", 1, (1, None),
                 ("triangle", "tetrahedron"))

register_element("Nedelec 2nd kind H(curl)", "N2curl", 1, (1, None),
                 ("triangle", "tetrahedron"))

register_element("Quadrature", "Q", 0, (0, None),
                 (None, "interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))

register_element("Boundary Quadrature", "BQ", 0, (0, None),
                 (None, "interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))

register_element("Real", "R", 0, (0, 0),
                 (None, "interval", "triangle", "tetrahedron", "quadrilateral", "hexahedron"))

# Let Nedelec H(div) elements be aliases to BDMs/RTs
register_alias("Nedelec 1st kind H(div)",
               lambda family, cell, order, degree: ("Raviart-Thomas", cell, order))
register_alias("N1div",
               lambda family, cell, order, degree: ("Raviart-Thomas", cell, order))

register_alias("Nedelec 2nd kind H(div)",
               lambda family, cell, order, degree: ("Brezzi-Douglas-Marini", cell, order))
register_alias("N2div",
               lambda family, cell, order, degree: ("Brezzi-Douglas-Marini", cell, order))

# Use P Lambda/P- Lambda aliases
register_alias("P Lambda", lambda *foo: FEEC_aliases(*foo))
register_alias("P- Lambda", lambda *foo: FEEC_aliases(*foo))



