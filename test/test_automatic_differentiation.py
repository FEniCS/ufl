#!/usr/bin/env python

"""
These tests should cover the behaviour of the automatic differentiation
algorithm at a technical level, and are thus implementation specific.
Other tests check for mathematical correctness of diff and derivative.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

from ufl.classes import Grad
#from ufl.algorithms import expand_derivatives2
from ufl.algorithms import expand_derivatives

class ForwardADTestCase(UflTestCase):

    def setUp(self):
        super(ForwardADTestCase, self).setUp()

    def tearDown(self):
        super(ForwardADTestCase, self).tearDown()

    def ad_algorithm(self, expr):
        return expand_derivatives(expr)
        #return expand_derivatives2(expr)

    def test_when_there_are_no_derivatives_the_expression_does_not_change(self):
        cell = cell2D

        U = FiniteElement("U", cell, None)
        V = VectorElement("U", cell, None)
        W = TensorElement("U", cell, None)

        u = Coefficient(U)
        v = Coefficient(V)
        w = Coefficient(W)

        du = Argument(U)
        dv = Argument(V)
        dw = Argument(W)

        d = cell.d
        x = cell.x
        n = cell.n
        c = cell.volume
        h = cell.circumradius
        f = cell.facet_area
        s = cell.surface_area

        I = Identity(d)

        i,j,k,l = indices(4)

        expressions = ([
            0,
            1,
            3.14,
            I,

            x,
            n,
            c,
            h,
            f,
            s,

            u,
            du,
            v,
            dv,
            w,
            dw,

            u*2,
            v*2,
            w*2,

            u+2*u,
            v+2*v,
            w+2*w,

            2/u,
            u/2,
            v/2,
            w/2,

            u**3,
            3**u,

            abs(u),
            sqrt(u),
            exp(u),
            ln(u),

            cos(u),
            sin(u),
            tan(u),
            acos(u),
            asin(u),
            atan(u),

            erf(u),
            bessel_I(1,u),
            bessel_J(1,u),
            bessel_K(1,u),
            ])

        for expr in expressions:
            before = as_ufl(expr)
            after = self.ad_algorithm(before)
            print '\n', str(before), '\n', str(after), '\n'
            self.assertEqual(before, after)

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()

