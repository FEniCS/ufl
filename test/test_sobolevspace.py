#!/usr/bin/env python

__authors__ = "David Ham"
__date__ = "2014-03-04"

from ufltestcase import UflTestCase, main
from ufl import FiniteElement, triangle
from ufl.sobolevspace import H2, H1, HDiv, HCurl, L2, SobolevSpace

class TestSobolevSpace(UflTestCase):

    def test_inclusion(self):

        assert H2 < H1       # Inclusion
        assert not H2 > H1   # Not included
        assert HDiv <= HDiv  # Reflexivity
        assert H2 < L2       # Transitivity

    def test_repr(self):
        assert eval(repr(H2)) == H2

    def xtest_contains(self):
        # TODO: Add sobolev_space() to elements
        assert FiniteElement("DG", triangle, 0) in L2
        assert FiniteElement("DG", triangle, 0) not in H1
        assert FiniteElement("CG", triangle, 1) in H1
        assert FiniteElement("CG", triangle, 1) in L2

if __name__ == "__main__":
    main()
