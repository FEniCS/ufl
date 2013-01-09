#!/usr/bin/env python

"""
Tests of domain language and attaching domains to forms.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

#from ufl.classes import ...
#from ufl.algorithms import ...



all_cells = (cell1D, cell2D, cell3D, interval, triangle, tetrahedron, quadrilateral, hexahedron)

class Domain(object):
    def __init__(self, cell, name="default"):
        self._cell = cell
        self._name = name

    def cell(self):
        return self._cell

    def name(self):
        return self._name

    def __eq__(self, other):
        return (isinstance(other, Domain)
                and self._cell == other._cell
                and self._name == other._name)

    def __repr__(self):
        return "Domain(%r, %r)" % (self._cell, self._name)

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self):
        return False

# Cells are mapped internally in UFL to a default domain for compatibility:
_default_domains = dict((cell, Domain(cell)) for cell in all_cells)

def as_domain(domain):
    if isinstance(domain, Domain):
        return domain
    elif isinstance(domain, Cell):
        return _default_domains[domain]
    else:
        error("Invalid domain %s." % str(domain))


class DomainTestCase(UflTestCase):

    def test_construct_domains_from_cells(self):
        for cell in all_cells:
            D1 = Domain(cell)
            D2 = as_domain(cell)
            self.assertFalse(D1 is D2)
            self.assertEqual(D1, D2)

    def test_as_domain_from_cell_is_unique(self):
        for cell in all_cells:
            D1 = as_domain(cell)
            D2 = as_domain(cell)
            self.assertTrue(D1 is D2)

    def test_construct_domains_with_names(self):
        for cell in all_cells:
            D2 = Domain(cell, name="D2")
            D3 = Domain(cell, name="D3")
            self.assertNotEqual(D2, D3)

    def test_domains_sort_by_name(self):
        domains = [Domain(cell, "D%d"%k) for (k,cell) in enumerate(all_cells)]
        #sdomains = sorted(domains)
        #self.assertEqual()


# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()
