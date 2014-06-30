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

        n = FacetNormal(triangle)

        a = u*v*dx
        L = f*v*dx
        b = u*v*dx(0) +inner(c*grad(u), grad(v))*dx(1) + dot(n, grad(u))*v*ds + f*v*dx

        self.elements = (element,)
        self.arguments = (v, u)
        self.coefficients = (c, f)
        self.forms = (a, L, b)

        if False:
            print()
            print((form_info(a)))
            print()
            print((form_info(L)))
            print()
            print((form_info(b)))
            print()

        if False:
            print()
            print([str(c) for c in self.coefficients])
            print()
            print((str(self.forms[2])))
            print()
            print([str(b) for b in extract_arguments(self.forms[2])])
            print()
            print((self.coefficients))
            print()
            print((repr(self.forms[2])))
            print()
            print((extract_arguments(self.forms[2])))
            print()

    def test_arguments(self):
        assert self.arguments == tuple(extract_arguments(self.forms[0]))
        assert tuple(self.arguments[:1]) == tuple(extract_arguments(self.forms[1]))

    def test_coefficients(self):
        assert self.coefficients == tuple(extract_coefficients(self.forms[2]))

    def test_elements(self):
        #print elements(self.forms[2])
        #print unique_elements(self.forms[2])
        #print unique_classes(self.forms[2])
        b = self.forms[2]
        integrals = b.integrals_by_type(Measure.CELL)
        integrand = integrals[0].integrand()
        d = extract_duplications(integrand)
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

        for itg in a.integrals_by_type(Measure.CELL):
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

    def test_adjoint(self):
        cell = triangle

        V1 = FiniteElement("CG", cell, 1)
        V2 = FiniteElement("CG", cell, 2)

        u = TrialFunction(V1)
        v = TestFunction(V2)
        self.assertGreater(u.number(), v.number())

        u2 = Argument(V1, 2)
        v2 = Argument(V2, 3)
        self.assertLess(u2.number(), v2.number())

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
