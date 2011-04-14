#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2011-04-14 -- 2011-04-14"

from ufltestcase import UflTestCase, main

from ufl import *
from ufl.classes import Indexed

class LiteralsTestCase(UflTestCase):

    def test_permutation_symbol_3(self):
        e = PermutationSymbol(3)
        self.assertEqual(e.shape(), (3, 3, 3))
        self.assertEqual(eval(repr(e)), e)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    value = (j-i)*(k-i)*(k-j)/2
                    self.assertEqual(e[i, j, k], value)
        i, j, k = indices(3)
        self.assertIsInstance(e[i,j,k], Indexed)
        x = (0,0,0)
        self.assertEqual((e[i,j,k] * e[i,j,k])(x), 6)

if __name__ == "__main__":
    main()
