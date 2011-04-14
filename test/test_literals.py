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

    def test_permutation_symbol_n(self):
        for n in range(2,5): # tested with upper limit 7, but evaluation is a bit slow then
            e = PermutationSymbol(n)
            self.assertEqual(e.shape(), (n,)*n)
            self.assertEqual(eval(repr(e)), e)

            ii = indices(n)
            x = (0,)*n
            nfac = product(m for m in range(1,n+1))
            self.assertEqual((e[ii] * e[ii])(x), nfac)

if __name__ == "__main__":
    main()
