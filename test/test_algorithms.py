#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-12 -- 2009-01-28"

# Modified by Anders Logg, 2008
# Modified by Garth N. Wells, 2009

from ufltestcase import UflTestCase, main
from pprint import *

from ufl import *
from ufl.algorithms import *
from ufl.classes import Sum, Product

# TODO: add more tests, covering all utility algorithms

class AlgorithmsTestCase(UflTestCase):

    def setUp(self):
        super(AlgorithmsTestCase, self).setUp()

        element = FiniteElement("CG", triangle, 1)

        v = TestFunction(element)
        u = TrialFunction(element)

        c = Coefficient(element)
        f = Coefficient(element)

        n = triangle.n

        a = u*v*dx
        L = f*v*dx
        b = u*v*dx(0) +inner(c*grad(u),grad(v))*dx(1) + dot(n, grad(u))*v*ds + f*v*dx

        self.elements = (element,)
        self.arguments = (v, u)
        self.coefficients = (c, f)
        self.forms = (a, L, b)

        if False:
            print
            print form_info(a)
            print
            print form_info(L)
            print
            print form_info(b)
            print

        if False:
            print
            print [str(c) for c in self.coefficients]
            print
            print str(self.forms[2])
            print
            print [str(b) for b in extract_arguments(self.forms[2])]
            print
            print self.coefficients
            print
            print repr(self.forms[2])
            print
            print extract_arguments(self.forms[2])
            print

    def tearDown(self):
        super(AlgorithmsTestCase, self).tearDown()

    def test_flatten(self):
        element = FiniteElement("CG", "triangle", 1)
        a = Coefficient(element)
        b = Coefficient(element)
        c = Coefficient(element)
        d = Coefficient(element)

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

    def test_arguments(self):
        assert self.arguments == tuple(extract_arguments(self.forms[0]))
        assert tuple(self.arguments[:1]) == tuple(extract_arguments(self.forms[1]))

    def test_coefficients(self):
        assert self.coefficients == tuple(extract_coefficients(self.forms[2]))

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
        self.assertEqual((element1, element2), extract_elements(a))
        self.assertEqual((element1,), extract_unique_elements(a))

    def test_walk(self):
        element = FiniteElement("CG", "triangle", 1)
        v = TestFunction(element)
        f = Coefficient(element)
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

        self.assertEqual(prestore, [(p, 0), (v, 1), (f, 1)]) # NB! Sensitive to ordering of expressions.
        self.assertEqual(poststore, [(v, 1), (f, 1), (p, 0)]) # NB! Sensitive to ordering of expressions.
        #print "\n"*2 + "\n".join(map(str,prestore))
        #print "\n"*2 + "\n".join(map(str,poststore))

    def test_traversal(self):
        element = FiniteElement("CG", "triangle", 1)
        v = TestFunction(element)
        f = Coefficient(element)
        g = Coefficient(element)
        p1 = f*v
        p2 = g*v
        s = p1 + p2
        pre_traverse = list(pre_traversal(s))
        post_traverse = list(post_traversal(s))

        self.assertEqual(pre_traverse, [s, p1, v, f, p2, v, g]) # NB! Sensitive to ordering of expressions.
        self.assertEqual(post_traverse, [v, f, p1, v, g, p2, s]) # NB! Sensitive to ordering of expressions.

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

    def test_max_degree_estimation(self):
        V1 = FiniteElement("CG", triangle, 1)
        V2 = FiniteElement("CG", triangle, 2)
        VV = VectorElement("CG", triangle, 3)
        VM = V1 * V2
        v1 = Argument(V1)
        v2 = Argument(V2)
        f1, f2 = Coefficients(VM)
        vv = Argument(VV)
        vu = Argument(VV)

        self.assertEqual(estimate_max_polynomial_degree(vv[0]), 3)
        self.assertEqual(estimate_max_polynomial_degree(v2*vv[0]), 3)
        self.assertEqual(estimate_max_polynomial_degree(vu[0]*vv[0]), 3)
        self.assertEqual(estimate_max_polynomial_degree(vu[i]*vv[i]), 3)

        self.assertEqual(estimate_max_polynomial_degree(v1), 1)
        self.assertEqual(estimate_max_polynomial_degree(v2), 2)

        # TODO: This should be 1, but 2 is expected behaviour now
        # because f1 is part of a mixed element with max degree 2.
        self.assertEqual(estimate_max_polynomial_degree(f1), 2)

        self.assertEqual(estimate_max_polynomial_degree(f2), 2)
        self.assertEqual(estimate_max_polynomial_degree(v2*v1), 2)

        # TODO: This should be 1, but 2 is expected behaviour now
        # because f1 is part of a mixed element with max degree 2.
        self.assertEqual(estimate_max_polynomial_degree(f1*v1), 2)

        self.assertEqual(estimate_max_polynomial_degree(f2*v1), 2)
        self.assertEqual(estimate_max_polynomial_degree(f2*v2*v1), 2)
        self.assertEqual(estimate_max_polynomial_degree(f1*f2*v2.dx(0)*v1.dx(0)), 2)
        self.assertEqual(estimate_max_polynomial_degree(f2**3*v1 + f1*v1), 2)

    def test_total_degree_estimation(self):
        V1 = FiniteElement("CG", triangle, 1)
        V2 = FiniteElement("CG", triangle, 2)
        VV = VectorElement("CG", triangle, 3)
        VM = V1 * V2
        v1 = Argument(V1)
        v2 = Argument(V2)
        f1, f2 = Coefficients(VM)
        vv = Argument(VV)
        vu = Argument(VV)

        x, y = triangle.x
        self.assertEqual(estimate_total_polynomial_degree(x), 1)
        self.assertEqual(estimate_total_polynomial_degree(x*y), 2)
        self.assertEqual(estimate_total_polynomial_degree(x**3), 3)
        self.assertEqual(estimate_total_polynomial_degree(x**3), 3)
        self.assertEqual(estimate_total_polynomial_degree((x-1)**4), 4)

        self.assertEqual(estimate_total_polynomial_degree(vv[0]), 3)
        self.assertEqual(estimate_total_polynomial_degree(v2*vv[0]), 5)
        self.assertEqual(estimate_total_polynomial_degree(vu[0]*vv[0]), 6)
        self.assertEqual(estimate_total_polynomial_degree(vu[i]*vv[i]), 6)

        self.assertEqual(estimate_total_polynomial_degree(v1), 1)
        self.assertEqual(estimate_total_polynomial_degree(v2), 2)

        # TODO: This should be 1, but 2 is expected behaviour now
        # because f1 is part of a mixed element with max degree 2.
        self.assertEqual(estimate_total_polynomial_degree(f1), 2)

        self.assertEqual(estimate_total_polynomial_degree(f2), 2)
        self.assertEqual(estimate_total_polynomial_degree(v2*v1), 3)

        # TODO: This should be 2, but 3 is expected behaviour now
        # because f1 is part of a mixed element with max degree 2.
        self.assertEqual(estimate_total_polynomial_degree(f1*v1), 3)

        self.assertEqual(estimate_total_polynomial_degree(f2*v1), 3)
        self.assertEqual(estimate_total_polynomial_degree(f2*v2*v1), 5)

        self.assertEqual(estimate_total_polynomial_degree(f2+3), 2)
        self.assertEqual(estimate_total_polynomial_degree(f2*3), 2)
        self.assertEqual(estimate_total_polynomial_degree(f2**3), 6)
        self.assertEqual(estimate_total_polynomial_degree(f2/3), 2)
        self.assertEqual(estimate_total_polynomial_degree(f2/v2), 4)
        self.assertEqual(estimate_total_polynomial_degree(f2/(x-1)), 3)

        self.assertEqual(estimate_total_polynomial_degree(v1.dx(0)), 0)
        self.assertEqual(estimate_total_polynomial_degree(f2.dx(0)), 1)

        self.assertEqual(estimate_total_polynomial_degree(f2*v2.dx(0)*v1.dx(0)), 2+1)
        self.assertEqual(estimate_total_polynomial_degree(f2**3*v1 + f1*v1), 7)

        # Based on the arbitrary chosen math function heuristics...
        nx, ny = triangle.n
        self.assertEqual(estimate_total_polynomial_degree(sin(nx**2)), 0)
        self.assertEqual(estimate_total_polynomial_degree(sin(x**3)), 3+2)

    def test_adjoint(self):
        cell = triangle

        V1 = FiniteElement("CG", cell, 1)
        V2 = FiniteElement("CG", cell, 2)

        u = TrialFunction(V1)
        v = TestFunction(V2)
        self.assertGreater(u.count(), v.count())

        u2 = Argument(V1)
        v2 = Argument(V2)
        self.assertLess(u2.count(), v2.count())

        a = u*v*dx
        a_arg_degrees = [arg.element().degree() for arg in extract_arguments(a)]
        self.assertEqual(a_arg_degrees, [2, 1])

        b = adjoint(a)
        b_arg_degrees = [arg.element().degree() for arg in extract_arguments(b)]
        self.assertEqual(b_arg_degrees, [1, 2])

        c = adjoint(a, (u2, v2))
        c_arg_degrees = [arg.element().degree() for arg in extract_arguments(c)]
        self.assertEqual(c_arg_degrees, [1, 2])

        d = adjoint(b)
        d_arg_degrees = [arg.element().degree() for arg in extract_arguments(d)]
        self.assertEqual(d_arg_degrees, [2, 1])

if __name__ == "__main__":
    main()
