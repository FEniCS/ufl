#!/usr/bin/env python

"""
Test use of grad in various situations.
"""

# These are thin wrappers on top of unittest.TestCase and unittest.main
from ufltestcase import UflTestCase, main

# This imports everything external code will see from ufl
from ufl import *

#from ufl.classes import ...
from ufl.algorithms import compute_form_data


class GradTestCase(UflTestCase):

    def xtest_grad_div_curl_properties_in_1D(self):
       self._test_grad_div_curl_properties(interval)

    def test_grad_div_curl_properties_in_2D(self):
       self._test_grad_div_curl_properties(cell2D)

    def xtest_grad_div_curl_properties_in_3D(self):
       self._test_grad_div_curl_properties(cell3D)

    def _test_grad_div_curl_properties(self, cell):
        d = cell.geometric_dimension()

        S = FiniteElement("CG", cell, 1)
        V = VectorElement("CG", cell, 1)
        T = TensorElement("CG", cell, 1)

        cs = Constant(cell)
        cv = VectorConstant(cell)
        ct = TensorConstant(cell)

        s = Coefficient(S)
        v = Coefficient(V)
        t = Coefficient(T)

        def eval_s(x, derivatives=()):
            return sum(derivatives)
        def eval_v(x, derivatives=()):
            return tuple(float(k)+sum(derivatives) for k in range(d))
        def eval_t(x, derivatives=()):
            return tuple(tuple(float(i*j)+sum(derivatives)
                               for i in range(d))
                               for j in range(d))

        mapping = { cs: eval_s, s: eval_s,
                    cv: eval_v, v: eval_v,
                    ct: eval_t, t: eval_t, }
        x = tuple(1.0+float(k) for k in range(d))

        self.assertEqual(s.ufl_shape, ())
        self.assertEqual(v.ufl_shape, (d,))
        self.assertEqual(t.ufl_shape, (d, d))

        self.assertEqual(cs.ufl_shape, ())
        self.assertEqual(cv.ufl_shape, (d,))
        self.assertEqual(ct.ufl_shape, (d, d))

        self.assertEqual(s(x, mapping=mapping), eval_s(x))
        self.assertEqual(v(x, mapping=mapping), eval_v(x))
        self.assertEqual(t(x, mapping=mapping), eval_t(x))

        self.assertEqual(grad(s).ufl_shape, (d,))
        self.assertEqual(grad(v).ufl_shape, (d, d))
        self.assertEqual(grad(t).ufl_shape, (d, d, d))

        self.assertEqual(grad(cs).ufl_shape, (d,))
        self.assertEqual(grad(cv).ufl_shape, (d, d))
        self.assertEqual(grad(ct).ufl_shape, (d, d, d))

        self.assertEqual(grad(s)[0](x, mapping=mapping), eval_s(x, (0,)))
        self.assertEqual(grad(v)[d-1, d-1](x, mapping=mapping),
                         eval_v(x, derivatives=(d-1,))[d-1])
        self.assertEqual(grad(t)[d-1, d-1, d-1](x, mapping=mapping),
                         eval_t(x, derivatives=(d-1,))[d-1][d-1])

        self.assertEqual(div(grad(cs)).ufl_shape, ())
        self.assertEqual(div(grad(cv)).ufl_shape, (d,))
        self.assertEqual(div(grad(ct)).ufl_shape, (d, d))

        self.assertEqual(s.dx(0).ufl_shape, ())
        self.assertEqual(v.dx(0).ufl_shape, (d,))
        self.assertEqual(t.dx(0).ufl_shape, (d, d))

        self.assertEqual(s.dx(0, 0).ufl_shape, ())
        self.assertEqual(v.dx(0, 0).ufl_shape, (d,))
        self.assertEqual(t.dx(0, 0).ufl_shape, (d, d))

        i, j = indices(2)
        self.assertEqual(s.dx(i).ufl_shape, ())
        self.assertEqual(v.dx(i).ufl_shape, (d,))
        self.assertEqual(t.dx(i).ufl_shape, (d, d))

        self.assertEqual(s.dx(i).free_indices(), (i,))
        self.assertEqual(v.dx(i).free_indices(), (i,))
        self.assertEqual(t.dx(i).free_indices(), (i,))

        self.assertEqual(s.dx(i, j).ufl_shape, ())
        self.assertEqual(v.dx(i, j).ufl_shape, (d,))
        self.assertEqual(t.dx(i, j).ufl_shape, (d, d))

        # This comparison is unstable w.r.t. sorting of i,j
        self.assertTrue(s.dx(i, j).free_indices() in [(i, j), (j, i)])
        self.assertTrue(v.dx(i, j).free_indices() in [(i, j), (j, i)])
        self.assertTrue(t.dx(i, j).free_indices() in [(i, j), (j, i)])

        a0 = s.dx(0)*dx
        a1 = s.dx(0)**2*dx
        a2 = v.dx(0)**2*dx
        a3 = t.dx(0)**2*dx

        a4 = inner(grad(s), grad(s))*dx
        a5 = inner(grad(v), grad(v))*dx
        a6 = inner(grad(t), grad(t))*dx

        a7 = inner(div(grad(s)), s)*dx
        a8 = inner(div(grad(v)), v)*dx
        a9 = inner(div(grad(t)), t)*dx

        fd0 = compute_form_data(a0)
        fd1 = compute_form_data(a1)
        fd2 = compute_form_data(a2)
        fd3 = compute_form_data(a3)

        fd4 = compute_form_data(a4)
        fd5 = compute_form_data(a5)
        fd6 = compute_form_data(a6)

        fd7 = compute_form_data(a7)
        fd8 = compute_form_data(a8)
        fd9 = compute_form_data(a9)

        #self.assertTrue(False) # Just to show it runs

# Don't touch these lines, they allow you to run this file directly
if __name__ == "__main__":
    main()
