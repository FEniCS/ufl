#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-12 -- 2009-01-28"

# Modified by Anders Logg, 2008
# Modified by Garth N. Wells, 2009

import unittest
from pprint import *

from ufl import *
from ufl.algorithms import * 
from ufl.classes import Sum, Product

# TODO: add more tests, covering all utility algorithms

class AlgorithmsTestCase(unittest.TestCase):

    def setUp(self):
        element = FiniteElement("CG", triangle, 1)
        
        v = TestFunction(element)
        u = TrialFunction(element)
        
        c = Function(element)
        f = Function(element)
        
        n = triangle.n
        
        a = u*v*dx
        L = f*v*dx
        b = u*v*dx(0) +inner(c*grad(u),grad(v))*dx(1) + dot(n, grad(u))*v*ds + f*v*dx
        
        self.elements = (element,)
        self.basis_functions = (v, u)
        self.functions = (c, f)
        self.forms = (a, L, b)

        if False:
            print
            print
            print
            print form_info(a)
            print
            print
            print
            print form_info(L)
            print
            print
            print
            print form_info(b)
            print
            print
            print
        
        if False:
            print 
            print [str(c) for c in self.functions]
            print 
            print str(self.forms[2])
            print 
            print [str(b) for b in extract_basis_functions(self.forms[2])]
            print 
            print self.functions
            print 
            print repr(self.forms[2])
            print 
            print extract_basis_functions(self.forms[2])
            print 
    
    def test_flatten(self):
        element = FiniteElement("CG", "triangle", 1)
        a = Function(element)
        b = Function(element)
        c = Function(element)
        d = Function(element)
        
        a  = (a+b)+(c+d)
        fa = flatten(a)
        assert isinstance(a,  Sum) and len(a.operands())  == 2
        assert isinstance(fa, Sum) and len(fa.operands()) == 4
        aa, ab = a.operands()
        assert isinstance(aa, Sum) and len(aa.operands()) == 2
        assert isinstance(ab, Sum) and len(ab.operands()) == 2
        
        a  = (a*b)*(c*d)
        fa = flatten(a)
        assert isinstance(a,  Product) and len(a.operands())  == 2
        assert isinstance(fa, Product) and len(fa.operands()) == 4
        aa, ab = a.operands()
        assert isinstance(aa, Product) and len(aa.operands()) == 2
        assert isinstance(ab, Product) and len(ab.operands()) == 2

    def test_basis_functions(self):
        assert self.basis_functions == tuple(extract_basis_functions(self.forms[0]))
        assert tuple(self.basis_functions[:1]) == tuple(extract_basis_functions(self.forms[1]))

    def test_functions(self):
        assert self.functions == tuple(extract_functions(self.forms[2]))

    def test_elements(self):
        #print elements(self.forms[2])
        #print unique_elements(self.forms[2])
        #print unique_classes(self.forms[2])
        d = extract_duplications(self.forms[2].cell_integrals()[0]._integrand)
        #pprint(list(d))

        element1 = FiniteElement("CG", triangle, 1)
        element2 = FiniteElement("CG", triangle, 1)

        v = TestFunction(element1)
        u = TrialFunction(element2)

        a = u*v*dx
        self.assertTrue( (element1, element2) == extract_elements(a) )
        self.assertTrue( set([element1]) == extract_unique_elements(a) )

    def test_walk(self):
        element = FiniteElement("CG", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        p = f*v
        a = p*dx

        prestore = []
        def pre(o, stack):
            prestore.append((o, len(stack)))
        poststore = []
        def post(o, stack):
            poststore.append((o, len(stack)))
        
        for itg in a.cell_integrals():
            walk(itg.integrand(), pre, post)
        
        self.assertTrue(prestore == [(p, 0), (v, 1), (f, 1)]) # NB! Sensitive to ordering of expressions.
        self.assertTrue(poststore == [(v, 1), (f, 1), (p, 0)]) # NB! Sensitive to ordering of expressions.
        #print "\n"*2 + "\n".join(map(str,prestore))
        #print "\n"*2 + "\n".join(map(str,poststore))

    def test_traversal(self):
        element = FiniteElement("CG", "triangle", 1)
        v = TestFunction(element)
        f = Function(element)
        g = Function(element)
        p1 = f*v
        p2 = g*v
        s = p1 + p2
        pre_traverse = list(pre_traversal(s))
        post_traverse = list(post_traversal(s))
        
        self.assertTrue(pre_traverse  == [s, p1, v, f, p2, v, g]) # NB! Sensitive to ordering of expressions.
        self.assertTrue(post_traverse == [v, f, p1, v, g, p2, s]) # NB! Sensitive to ordering of expressions.

    def test_expand_indices(self):
        element = FiniteElement("Lagrange", triangle, 2)
        v = TestFunction(element)
        u = TrialFunction(element)
        
        def evaluate(form):
            return form.cell_integral()[0].integrand()((), { v: 3, u: 5 }) # TODO: How to define values of derivatives?
        
        a = div(grad(v))*u*dx
        #a1 = evaluate(a)
        a = expand_derivatives(a)
        #a2 = evaluate(a)
        a = expand_indices(a)
        #a3 = evaluate(a)
        # TODO: Compare a1, a2, a3
        # TODO: Test something more

    def test_degree_estimation(self):
        V1 = FiniteElement("CG", triangle, 1)
        V2 = FiniteElement("CG", triangle, 2)
        VV = VectorElement("CG", triangle, 3)
        VM = V1 + V2
        v1 = BasisFunction(V1)
        v2 = BasisFunction(V2)
        f1, f2 = Functions(VM)
        vv = BasisFunction(VV)
        vu = BasisFunction(VV)
    
        self.assertEqual(estimate_max_polynomial_degree(vv[0]), 3)
        self.assertEqual(estimate_max_polynomial_degree(v2*vv[0]), 3)
        self.assertEqual(estimate_max_polynomial_degree(vu[0]*vv[0]), 3)
        self.assertEqual(estimate_max_polynomial_degree(vu[i]*vv[i]), 3)

        self.assertEqual(estimate_max_polynomial_degree(v1), 1)
        self.assertEqual(estimate_max_polynomial_degree(v2), 2)
        self.assertEqual(estimate_max_polynomial_degree(f1), 2) # TODO: This should be 1, but 2 is expected behaviour now.
        self.assertEqual(estimate_max_polynomial_degree(f2), 2)
        self.assertEqual(estimate_max_polynomial_degree(v2*v1), 2)
        self.assertEqual(estimate_max_polynomial_degree(f1*v1), 2) # TODO: This should be 1, but 2 is expected behaviour now.
        self.assertEqual(estimate_max_polynomial_degree(f2*v1), 2)
        self.assertEqual(estimate_max_polynomial_degree(f2*v2*v1), 2)
        self.assertEqual(estimate_max_polynomial_degree(f1*f2*v2.dx(0)*v1.dx(0)), 2)
        self.assertEqual(estimate_max_polynomial_degree(f2**3*v1 + f1*v1), 2)


tests = [AlgorithmsTestCase]

if __name__ == "__main__":
    unittest.main()
