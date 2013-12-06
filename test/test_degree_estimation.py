#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2008-03-12 -- 2009-01-28"

from ufltestcase import UflTestCase, main
from pprint import *

from ufl import *
from ufl.algorithms import *

class TestDegreeEstimation(UflTestCase):

    def test_total_degree_estimation(self):
        V1 = FiniteElement("CG", triangle, 1)
        V2 = FiniteElement("CG", triangle, 2)
        VV = VectorElement("CG", triangle, 3)
        VM = V1 * V2
        O1 = OuterProductElement(V1, V1)
        O2 = OuterProductElement(V2, V1)
        v1 = Argument(V1, 2)
        v2 = Argument(V2, 3)
        f1, f2 = Coefficients(VM)
        u1 = Coefficient(O1)
        u2 = Coefficient(O2)
        vv = Argument(VV, 4)
        vu = Argument(VV, 5)

        x, y = SpatialCoordinate(triangle)
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

        self.assertEqual(estimate_total_polynomial_degree(f2), 2)
        self.assertEqual(estimate_total_polynomial_degree(f2**2), 4)
        self.assertEqual(estimate_total_polynomial_degree(f2**3), 6)
        self.assertEqual(estimate_total_polynomial_degree(f2**3*v1), 7)
        self.assertEqual(estimate_total_polynomial_degree(f2**3*v1 + f1*v1), 7)

        # outer product tuple-degree tests
        self.assertEqual(estimate_total_polynomial_degree(u1), (1, 1))
        self.assertEqual(estimate_total_polynomial_degree(u2), (2, 1))
        # derivatives should do nothing (don't know in which direction they act)
        self.assertEqual(estimate_total_polynomial_degree(grad(u2)), (2, 1))
        self.assertEqual(estimate_total_polynomial_degree(u1*u1), (2, 2))
        self.assertEqual(estimate_total_polynomial_degree(u2*u1), (3, 2))
        self.assertEqual(estimate_total_polynomial_degree(u2*u2), (4, 2))
        self.assertEqual(estimate_total_polynomial_degree(u1**3), (3, 3))
        self.assertEqual(estimate_total_polynomial_degree(u1**3 + u2*u2), (4, 3))
        self.assertEqual(estimate_total_polynomial_degree(u2**2*u1), (5, 3))

        # Based on the arbitrary chosen math function heuristics...
        nx, ny = FacetNormal(triangle)
        self.assertEqual(estimate_total_polynomial_degree(sin(nx**2)), 0)
        self.assertEqual(estimate_total_polynomial_degree(sin(x**3)), 3+2)

    def test_some_compound_types(self):

        # NB! Although some compound types are supported here,
        # some derivatives and compounds must be preprocessed
        # prior to degree estimation. In generic code, this algorithm 
        # should only be applied after preprocessing.
        
        etpd = estimate_total_polynomial_degree

        P2 = FiniteElement("CG", triangle, 2)
        V2 = VectorElement("CG", triangle, 2)

        u = Coefficient(P2)
        v = Coefficient(V2)

        self.assertEqual(etpd(u.dx(0)), 2-1)
        self.assertEqual(etpd(grad(u)), 2-1)
        self.assertEqual(etpd(nabla_grad(u)), 2-1)
        self.assertEqual(etpd(div(u)), 2-1)

        self.assertEqual(etpd(v.dx(0)), 2-1)
        self.assertEqual(etpd(grad(v)), 2-1)
        self.assertEqual(etpd(nabla_grad(v)), 2-1)
        self.assertEqual(etpd(div(v)), 2-1)
        self.assertEqual(etpd(nabla_div(v)), 2-1)

        self.assertEqual(etpd(dot(v, v)), 2+2)
        self.assertEqual(etpd(inner(v, v)), 2+2)
        
        self.assertEqual(etpd(dot(grad(u), grad(u))), 2-1 + 2-1)
        self.assertEqual(etpd(inner(grad(u), grad(u))), 2-1 + 2-1)

        self.assertEqual(etpd(dot(grad(v), grad(v))), 2-1 + 2-1)
        self.assertEqual(etpd(inner(grad(v), grad(v))), 2-1 + 2-1)

if __name__ == "__main__":
    main()
