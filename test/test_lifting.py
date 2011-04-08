#!/usr/bin/env python
from ufltestcase import UflTestCase, main

from ufl import triangle, FiniteElement, VectorElement,\
    Coefficient, LiftingFunction, LiftingOperator,\
    TestFunction, dot, dx, dE
from ufl.algorithms import tree_format

class LiftingTestCase(UflTestCase):
    def test_lifting(self):
        doprint = False

        cell = triangle
        u_space = FiniteElement("DG", cell, 1)
        l_space = VectorElement("DG", cell, 0)
        R = LiftingFunction(l_space)
        r = LiftingOperator(l_space)
        
        u = Coefficient(u_space)
        v = TestFunction(u_space)
        
        a = dot(r(u), r(v))*dE
        if doprint:
            print 
            print str(a)
            print repr(a)
            print tree_format(a)
        
        a = dot(R(u), R(v))*dE
        if doprint:
            print 
            print str(a)
            print repr(a)
            print tree_format(a)

if __name__ == "__main__":
    main()
