#!/usr/bin/env python

__authors__ = "David Ham"
__date__ = "2014-03-04"

from ufltestcase import UflTestCase, main
from ufl.sobolevspace import H2, H1, HDiv, HCurl, L2

class TestSobolevSpace(UflTestCase):

    def test_inclusion(self):

        assert H2 < H1       # Inclusion
        assert not H2 > H1   # Not included
        assert HDiv <= HDiv  # Reflexivity
        assert H2 < L2       # Transitivity

if __name__ == "__main__":
    main()
