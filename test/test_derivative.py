#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-17 -- 2009-02-17"

from ufltestcase import UflTestCase, main
import math

from ufl import *
from ufl.constantvalue import as_ufl
from ufl.algorithms import expand_indices, strip_variables, post_traversal, preprocess

class DerivativeTestCase(UflTestCase):

    def setUp(self):
        super(DerivativeTestCase, self).setUp()
        self.cell = triangle
        self.element = FiniteElement("CG", self.cell, 1)
        self.v = TestFunction(self.element)
        self.u = TrialFunction(self.element)
        self.w = Coefficient(self.element)
        self.xv = ()
        self.uv = 7.0
        self.vv = 13.0
        self.wv = 11.0

    def _test(self, f, df):
        x = self.xv
        u, v, w = self.u, self.v, self.w
        mapping = { v: self.vv, u: self.uv, w: self.wv }

        dfv1 = derivative(f(w), w, v)
        dfv2 = df(w, v)
        dfv1 = dfv1(x, mapping)
        dfv2 = dfv2(x, mapping)
        self.assertEqual(dfv1, dfv2)

        dfv1 = derivative(f(7*w), w, v)
        dfv2 = 7*df(7*w, v)
        dfv1 = dfv1(x, mapping)
        dfv2 = dfv2(x, mapping)
        self.assertEqual(dfv1, dfv2)

    def testListTensor(self):
        v = variable(as_ufl(42))
        f = as_tensor((
                ( (0,      0), (0,   0) ),
                ( (v,    2*v), (0,   0) ),
                ( (v**2,   1), (2, v/2) ),
                ))
        self.assertEqual(f.shape(), (3,2,2))
        g = as_tensor((
                ( (0, 0), (0, 0) ),
                ( (1, 2), (0,0) ),
                ( (84, 0), (0, 0.5) ),
                ))
        self.assertEqual(g.shape(), (3,2,2))
        dfv = diff(f, v)
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    self.assertEqual(dfv[i,j,k](()), g[i,j,k](()))

    def testFunction(self):
        def f(w):  return w
        def df(w, v): return v
        self._test(f, df)

    def testSum(self):
        def f(w):  return w + 1
        def df(w, v): return v
        self._test(f, df)

    def testProduct(self):
        def f(w):  return 3*w
        def df(w, v): return 3*v
        self._test(f, df)

    def testPower(self):
        def f(w):  return w**3
        def df(w, v): return 3*w**2*v
        self._test(f, df)

    def testDivision(self):
        def f(w):  return w / 3.0
        def df(w, v): return v / 3.0
        self._test(f, df)

    def testDivision2(self):
        def f(w):  return 3.0 / w
        def df(w, v): return -3.0 * v / w**2
        self._test(f, df)

    def testExp(self):
        def f(w):  return exp(w)
        def df(w, v): return v*exp(w)
        self._test(f, df)

    def testLn(self):
        def f(w):  return ln(w)
        def df(w, v): return v / w
        self._test(f, df)

    def testCos(self):
        def f(w):  return cos(w)
        def df(w, v): return -v*sin(w)
        self._test(f, df)

    def testSin(self):
        def f(w):  return sin(w)
        def df(w, v): return v*cos(w)
        self._test(f, df)

    def testTan(self):
        def f(w):  return tan(w)
        def df(w, v): return v*2.0/(cos(2.0*w) + 1.0)
        self._test(f, df)

# TODO: Check the following tests. They run into strange math domain errors.
#     def testAcos(self):
#         def f(w):  return acos(w)
#         def df(w, v): return -v/sqrt(1.0 - w**2)
#         self._test(f, df)

#     def testAsin(self):
#         def f(w):  return asin(w)
#         def df(w, v): return v/sqrt(1.0 - w**2)
#         self._test(f, df)

    def testAtan(self):
        def f(w):  return atan(w)
        def df(w, v): return v/(1.0 + w**2)
        self._test(f, df)

    def testIndexSum(self):
        def f(w):
            # 3*w + 4*w**2 + 5*w**3
            a = as_vector((w, w**2, w**3))
            b = as_vector((3, 4, 5))
            i, = indices(1)
            return a[i]*b[i]
        def df(w, v): return 3*v + 4*2*w*v + 5*3*w**2*v
        self._test(f, df)

    def testHyperElasticity(self):
        cell = interval
        element = FiniteElement("CG", cell, 2)
        w = Coefficient(element)
        v = TestFunction(element)
        u = TrialFunction(element)
        b = Constant(cell)
        K = Constant(cell)

        dw = w.dx(0)
        dv = v.dx(0)
        du = v.dx(0)

        E = dw + dw**2 / 2
        E = variable(E)
        Q = b*E**2
        psi = K*(exp(Q)-1)

        f = psi*dx
        F = derivative(f, w, v)
        J = derivative(F, w, u)

        form_data_f = f.compute_form_data()
        form_data_F = F.compute_form_data()
        form_data_J = J.compute_form_data()

        f = form_data_f.preprocessed_form
        F = form_data_F.preprocessed_form
        J = form_data_J.preprocessed_form

        f_expression = strip_variables(f.cell_integrals()[0].integrand())
        F_expression = strip_variables(F.cell_integrals()[0].integrand())
        J_expression = strip_variables(J.cell_integrals()[0].integrand())

        #classes = set(c.__class__ for c in post_traversal(f_expression))

        Kv = .2
        bv = .3
        dw = .5
        dv = .7
        du = .11
        E = dw + dw**2 / 2.
        Q = bv*E**2
        expQ = float(exp(Q))
        psi = Kv*(expQ-1)
        fv = psi
        Fv = 2*Kv*bv*E*(1+dw)*expQ*dv
        Jv = 2*Kv*bv*expQ*dv*du*(E + (1+dw)**2*(2*bv*E**2 + 1))

        def Nv(x, derivatives):
            assert derivatives == (0,)
            return dv

        def Nu(x, derivatives):
            assert derivatives == (0,)
            return du

        def Nw(x, derivatives):
            assert derivatives == (0,)
            return dw

        w, b, K = form_data_f.coefficients
        mapping = { K: Kv, b: bv, w: Nw }
        fv2 = f_expression((0,), mapping)
        self.assertAlmostEqual(fv, fv2)

        w, b, K = form_data_F.coefficients
        v, = form_data_F.arguments
        mapping = { K: Kv, b: bv, v: Nv, w: Nw }
        Fv2 = F_expression((0,), mapping)
        self.assertAlmostEqual(Fv, Fv2)

        w, b, K = form_data_J.coefficients
        v, u = form_data_J.arguments
        mapping = { K: Kv, b: bv, v: Nv, u: Nu, w: Nw }
        Jv2 = J_expression((0,), mapping)
        self.assertAlmostEqual(Jv, Jv2)

    def test_mass_derived_from_functional(self):
	cell = triangle
        V = FiniteElement("CG", cell, 1)

        v = TestFunction(V)
        u = TrialFunction(V)
        w = Coefficient(V)

        f = (w**2/2)*dx
        L = w*v*dx
        a = u*v*dx
        F  = derivative(f, w, v)
        J1 = derivative(L, w, u)
        J2 = derivative(F, w, u)
	# TODO: assert something

    def test_coefficient_derivatives(self):
        V = FiniteElement("Lagrange", triangle, 1)
        dv = TestFunction(V)
        du = TrialFunction(V)
        u = Coefficient(V)
        f = Coefficient(V)
        g = Coefficient(V)
        df = Coefficient(V)
        dg = Coefficient(V)

        cd = { f: df, g: dg }
        F = f*g*dx
        J = derivative(F, u, du, cd)
        fd = J.compute_form_data()
        J2 = fd.preprocessed_form

        # TODO: This looks good, but we need to assert something sensible
        # TODO: Add tests covering more cases, in particular mixed stuff
        if 0:
            print
            print 'f ', f
            print 'df', df
            print 'g ', g
            print 'dg', dg
            print 'u ', u
            print map(str, fd.original_coefficients)
            print map(str, fd.coefficients)
            print
            print str(J2)
            print
            print repr(J2)
            print

    def test_foobar(self):
        element = VectorElement("Lagrange", triangle, 1)
        v = TestFunction(element)

        du = TrialFunction(element)

        U = Coefficient(element)

        def planarGrad(u):
            return as_matrix([[u[0].dx(0), 0 ,u[0].dx(1)],
                              [ 0 , 0 , 0 ],
                              [u[1].dx(0), 0 ,u[1].dx(1)]])

        def epsilon(u):
            return 0.5*(planarGrad(u)+planarGrad(u).T)

        def NS_a(u,v):
            return inner(epsilon(u),epsilon(v))

        L = NS_a(U,v)*dx
        a = derivative(L, U, du)
	# TODO: assert something

if __name__ == "__main__":
    main()

