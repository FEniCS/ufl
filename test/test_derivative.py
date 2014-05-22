#!/usr/bin/env python

__authors__ = "Martin Sandve Alnes"
__date__ = "2009-02-17 -- 2009-02-17"

from ufltestcase import UflTestCase, main
import math
from itertools import chain

from ufl import *
from ufl.constantvalue import as_ufl
from ufl.algorithms import expand_indices, strip_variables, post_traversal, compute_form_data, compute_form_signature

class DerivativeTestCase(UflTestCase):

    def setUp(self):
        super(DerivativeTestCase, self).setUp()
        self.cell = triangle
        self.element = FiniteElement("CG", self.cell, 1)
        self.v = TestFunction(self.element)
        self.u = TrialFunction(self.element)
        self.w = Coefficient(self.element)
        self.xv = (0.3, 0.7)
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

    # --- Literals

    def testScalarLiteral(self):
        def f(w):     return as_ufl(1)
        def df(w, v): return zero()
        self._test(f, df)

    def testIdentityLiteral(self):
        def f(w):     return Identity(2)[i,i]
        def df(w, v): return zero()
        self._test(f, df)

    # --- Form arguments

    def testCoefficient(self):
        def f(w):     return w
        def df(w, v): return v
        self._test(f, df)

    def testArgument(self):
        def f(w):     return self.v
        def df(w, v): return zero()
        self._test(f, df)

    # --- Geometry

    def testSpatialCoordinate(self):
        def f(w):     return SpatialCoordinate(triangle)[0]
        def df(w, v): return zero()
        self._test(f, df)

    def testFacetNormal(self):
        def f(w):     return FacetNormal(triangle)[0]
        def df(w, v): return zero()
        self._test(f, df)

    #def testCellSurfaceArea(self):
    #    def f(w):     return CellSurfaceArea(triangle)
    #    def df(w, v): return zero()
    #    self._test(f, df)

    def testFacetArea(self):
        def f(w):     return FacetArea(triangle)
        def df(w, v): return zero()
        self._test(f, df)

    def testCircumradius(self):
        def f(w):     return Circumradius(triangle)
        def df(w, v): return zero()
        self._test(f, df)

    def testCellVolume(self):
        def f(w):     return CellVolume(triangle)
        def df(w, v): return zero()
        self._test(f, df)

    # --- Basic operators

    def testSum(self):
        def f(w):     return w + 1
        def df(w, v): return v
        self._test(f, df)

    def testProduct(self):
        def f(w):     return 3*w
        def df(w, v): return 3*v
        self._test(f, df)

    def testPower(self):
        def f(w):     return w**3
        def df(w, v): return 3*w**2*v
        self._test(f, df)

    def testDivision(self):
        def f(w):     return w / 3.0
        def df(w, v): return v / 3.0
        self._test(f, df)

    def testDivision2(self):
        def f(w):     return 3.0 / w
        def df(w, v): return -3.0 * v / w**2
        self._test(f, df)

    def testExp(self):
        def f(w):     return exp(w)
        def df(w, v): return v*exp(w)
        self._test(f, df)

    def testLn(self):
        def f(w):     return ln(w)
        def df(w, v): return v / w
        self._test(f, df)

    def testCos(self):
        def f(w):     return cos(w)
        def df(w, v): return -v*sin(w)
        self._test(f, df)

    def testSin(self):
        def f(w):     return sin(w)
        def df(w, v): return v*cos(w)
        self._test(f, df)

    def testTan(self):
        def f(w):     return tan(w)
        def df(w, v): return v*2.0/(cos(2.0*w) + 1.0)
        self._test(f, df)

    def testAcos(self):
        def f(w):     return acos(w/1000)
        def df(w, v): return -(v/1000)/sqrt(1.0 - (w/1000)**2)
        self._test(f, df)

    def testAsin(self):
        def f(w):     return asin(w/1000)
        def df(w, v): return (v/1000)/sqrt(1.0 - (w/1000)**2)
        self._test(f, df)

    def testAtan(self):
        def f(w):     return atan(w)
        def df(w, v): return v/(1.0 + w**2)
        self._test(f, df)

    # FIXME: Add the new erf and bessel_*

    # --- Abs and conditionals

    def testAbs(self):
        def f(w):     return abs(w)
        def df(w, v): return sign(w)*v
        self._test(f, df)

    def testConditional(self):
        def cond(w): return lt(1.0, 2.0)
        def f(w):     return conditional(cond(w), 2*w, 3*w)
        def df(w, v): return 2*v
        self._test(f, df)

        def cond(w): return lt(2.0, 1.0)
        def f(w):     return conditional(cond(w), 2*w, 3*w)
        def df(w, v): return 3*v
        self._test(f, df)

    def testConditional(self): # This will fail without bugfix in derivative
        def cond(w): return lt(w, 1.0)
        def f(w):     return conditional(cond(w), 2*w, 3*w)
        def df(w, v): return (conditional(cond(w), 1, 0) * 2*v +
                              conditional(cond(w), 0, 1) * 3*v)
        self._test(f, df)

    # --- Tensor algebra basics

    def testIndexSum(self):
        def f(w):
            # 3*w + 4*w**2 + 5*w**3
            a = as_vector((w, w**2, w**3))
            b = as_vector((3, 4, 5))
            i, = indices(1)
            return a[i]*b[i]
        def df(w, v): return 3*v + 4*2*w*v + 5*3*w**2*v
        self._test(f, df)

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
        x = None
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    self.assertEqual(dfv[i,j,k](x), g[i,j,k](x))

    # --- Coefficient and argument input configurations

    def assertEqualExpr(self, a, b):
        a2 = compute_form_data(a*dx).preprocessed_form
        b2 = compute_form_data(b*dx).preprocessed_form
        if not a2 == b2:
            print
            print str(a2)
            print
            print str(b2)
            print
        self.assertEqual(a2, b2)

    def assertEqualBySampling(self, actual, expected):
        ad = compute_form_data(actual*dx)
        a = ad.preprocessed_form.integrals_by_type(Measure.CELL)[0].integrand()
        bd = compute_form_data(expected*dx)
        b = bd.preprocessed_form.integrals_by_type(Measure.CELL)[0].integrand()

        self.assertEqual([ad.function_replace_map[ac] for ac in ad.reduced_coefficients],
                         [bd.function_replace_map[bc] for bc in bd.reduced_coefficients])

        n = ad.num_coefficients
        def make_value(c):
            if isinstance(c, Coefficient):
                z = 0.3
                m = c.count()
            else:
                z = 0.7
                m = c.number()
            if c.shape() == ():
                return z * (0.1 + 0.9 * m / n)
            elif len(c.shape()) == 1:
                return tuple((z * (j + 0.1 + 0.9 * m / n) for j in range(c.shape()[0])))
            else:
                raise NotImplementedError("Tensor valued expressions not supported here.")

        amapping = dict((c, make_value(c)) for c in chain(ad.original_coefficients, ad.original_arguments))
        bmapping = dict((c, make_value(c)) for c in chain(bd.original_coefficients, bd.original_arguments))

        acell = actual.cell()
        bcell = expected.cell()
        self.assertEqual(acell, bcell)
        if acell.geometric_dimension() == 1:
            x = (0.3,)
        elif acell.geometric_dimension() == 2:
            x = (0.3, 0.4)
        elif acell.geometric_dimension() == 3:
            x = (0.3, 0.4, 0.5)
        av = a(x, amapping)
        bv = b(x, bmapping)

        if not av == bv:
            print "Tried to sample expressions to compare but failed:"
            print
            print str(a)
            print av
            print
            print str(b)
            print bv
            print

        self.assertEqual(av, bv)

    def test_single_scalar_coefficient_derivative(self):
        cell = triangle
        V = FiniteElement("CG", cell, 1)
        u = Coefficient(V)
        v = TestFunction(V)
        a = 3*u**2
        b = derivative(a, u, v)
        self.assertEqualExpr(b, 3*(u*(2*v)))

    def test_single_vector_coefficient_derivative(self):
        cell = triangle
        V = VectorElement("CG", cell, 1)
        u = Coefficient(V)
        v = TestFunction(V)
        a = 3*dot(u,u)
        actual = derivative(a, u, v)
        expected = 3*(2*(u[i]*v[i]))
        self.assertEqualBySampling(actual, expected)

    def test_multiple_coefficient_derivative(self):
        cell = triangle
        V = FiniteElement("CG", cell, 1)
        W = VectorElement("CG", cell, 1)
        M = V*W
        uv = Coefficient(V)
        uw = Coefficient(W)
        v = TestFunction(M)
        vv, vw = split(v)

        a = sin(uv)*dot(uw,uw)

        actual = derivative(a, (uv,uw), split(v))
        expected = cos(uv)*vv * (uw[i]*uw[i]) + (uw[j]*vw[j])*2 * sin(uv)
        self.assertEqualBySampling(actual, expected)

        actual = derivative(a, (uv,uw), v)
        expected = cos(uv)*vv * (uw[i]*uw[i]) + (uw[j]*vw[j])*2 * sin(uv)
        self.assertEqualBySampling(actual, expected)

    def test_indexed_coefficient_derivative(self):
        cell = triangle
        I = Identity(cell.geometric_dimension())
        V = FiniteElement("CG", cell, 1)
        W = VectorElement("CG", cell, 1)
        u = Coefficient(W)
        v = TestFunction(V)

        w = dot(u, nabla_grad(u))
        #a = dot(w, w)
        a = (u[i]*u[k].dx(i)) * w[k]

        actual = derivative(a, u[0], v)

        dw = v*u[k].dx(0) + u[i]*I[0,k]*v.dx(i)
        expected = 2 * w[k] * dw

        self.assertEqualBySampling(actual, expected)

    def test_multiple_indexed_coefficient_derivative(self):
        cell = tetrahedron
        I = Identity(cell.geometric_dimension())
        V = FiniteElement("CG", cell, 1)
        V2 = V*V
        W = VectorElement("CG", cell, 1)
        u = Coefficient(W)
        w = Coefficient(W)
        v = TestFunction(V2)
        vu, vw = split(v)

        actual = derivative(cos(u[i]*w[i]), (u[2], w[1]), (vu, vw))
        expected = -sin(u[i]*w[i])*(vu*w[2] + u[1]*vw)

        self.assertEqualBySampling(actual, expected)

    def test_segregated_derivative_of_convection(self):
        cell = tetrahedron
        V = FiniteElement("CG", cell, 1)
        W = VectorElement("CG", cell, 1)

        u = Coefficient(W)
        v = Coefficient(W)
        du = TrialFunction(V)
        dv = TestFunction(V)

        L = dot(dot(u, nabla_grad(u)), v)

        Lv = {}
        Lvu = {}
        for i in range(cell.geometric_dimension()):
            Lv[i] = derivative(L, v[i], dv)
            for j in range(cell.geometric_dimension()):
                Lvu[i,j] = derivative(Lv[i], u[j], du)

        for i in range(cell.geometric_dimension()):
            for j in range(cell.geometric_dimension()):
                form = Lvu[i,j]*dx
                fd = compute_form_data(form)
                pf = fd.preprocessed_form
                a = expand_indices(pf)
                #print (i,j), str(a)

        k = Index()
        for i in range(cell.geometric_dimension()):
            for j in range(cell.geometric_dimension()):
                actual = Lvu[i,j]
                expected = du*u[i].dx(j)*dv + u[k]*du.dx(k)*dv
                self.assertEqualBySampling(actual, expected)

    # --- User provided derivatives of coefficients

    def test_coefficient_derivatives(self):
        V = FiniteElement("Lagrange", triangle, 1)

        dv = TestFunction(V)

        f = Coefficient(V).reconstruct(count=0)
        g = Coefficient(V).reconstruct(count=1)
        df = Coefficient(V).reconstruct(count=2)
        dg = Coefficient(V).reconstruct(count=3)
        u = Coefficient(V).reconstruct(count=4)
        cd = { f: df, g: dg }

        integrand = inner(f, g)
        expected = (df*dv)*g + f*(dg*dv)

        F = integrand*dx
        J = derivative(F, u, dv, cd)
        fd = compute_form_data(J)
        actual = fd.preprocessed_form.integrals()[0].integrand()
        self.assertEqual(compute_form_signature(actual*dx), compute_form_signature(expected*dx))
        self.assertEqual(replace(actual, fd.function_replace_map), expected)

    def test_vector_coefficient_derivatives(self):
        V = VectorElement("Lagrange", triangle, 1)
        VV = TensorElement("Lagrange", triangle, 1)

        dv = TestFunction(V)

        df = Coefficient(VV).reconstruct(count=0)
        g = Coefficient(V).reconstruct(count=1)
        f = Coefficient(V).reconstruct(count=2)
        u = Coefficient(V).reconstruct(count=3)
        cd = { f: df }

        integrand = inner(f, g)

        i0, i1, i2, i3, i4 = [Index(count=c) for c in range(5)]
        expected = as_tensor(df[i2,i1]*dv[i1], (i2,))[i0]*g[i0]

        F = integrand*dx
        J = derivative(F, u, dv, cd)
        fd = compute_form_data(J)
        actual = fd.preprocessed_form.integrals()[0].integrand()
        self.assertEqual(compute_form_signature(actual*dx), compute_form_signature(expected*dx))
        #self.assertEqual(replace(actual, fd.function_replace_map), expected)

    def test_vector_coefficient_derivatives_of_product(self):
        V = VectorElement("Lagrange", triangle, 1)
        VV = TensorElement("Lagrange", triangle, 1)

        dv = TestFunction(V)

        df = Coefficient(VV).reconstruct(count=0)
        g = Coefficient(V).reconstruct(count=1)
        dg = Coefficient(VV).reconstruct(count=2)
        f = Coefficient(V).reconstruct(count=3)
        u = Coefficient(V).reconstruct(count=4)
        cd = { f: df, g: dg }

        integrand = f[i]*g[i]

        i0, i1, i2, i3, i4 = [Index(count=c) for c in range(5)]
        expected = as_tensor(df[i2,i1]*dv[i1], (i2,))[i0]*g[i0] +\
                   f[i0]*as_tensor(dg[i4,i3]*dv[i3], (i4,))[i0]

        F = integrand*dx
        J = derivative(F, u, dv, cd)
        fd = compute_form_data(J)
        actual = fd.preprocessed_form.integrals()[0].integrand()

        # Keeping this snippet here for a while for debugging purposes
        if 0:
            print '\n', 'str:'
            print str(actual)
            print str(expected)
            print '\n', 'repr:'
            print repr(actual)
            print repr(expected)
            from ufl.algorithms import tree_format
            open('actual.txt','w').write(tree_format(actual))
            open('expected.txt', 'w').write(tree_format(expected))
            print '\n', 'equal:'
            print str(actual) == str(expected)
            print repr(actual) == repr(expected)
            print actual == expected

        # Tricky case! These are equal in representation except
        # that the outermost sum/indexsum are swapped.
        # Sampling the expressions instead of comparing representations.
        x = (0, 0)
        funcs = {dv: (13, 14), f: (1,2), g: (3,4), df: ((5,6),(7,8)), dg: ((9,10),(11,12))}
        self.assertEqual(replace(actual,fd.function_replace_map)(x, funcs), expected(x, funcs))

        # TODO: Add tests covering more cases, in particular mixed stuff

    # --- Some actual forms

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

        form_data_f = compute_form_data(f)
        form_data_F = compute_form_data(F)
        form_data_J = compute_form_data(J)

        f = form_data_f.preprocessed_form
        F = form_data_F.preprocessed_form
        J = form_data_J.preprocessed_form

        f_expression = strip_variables(f.integrals_by_type(Measure.CELL)[0].integrand())
        F_expression = strip_variables(F.integrals_by_type(Measure.CELL)[0].integrand())
        J_expression = strip_variables(J.integrals_by_type(Measure.CELL)[0].integrand())

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

        w, b, K = form_data_f.original_coefficients
        mapping = { K: Kv, b: bv, w: Nw }
        fv2 = f_expression((0,), mapping)
        self.assertAlmostEqual(fv, fv2)

        w, b, K = form_data_F.original_coefficients
        v, = form_data_F.original_arguments
        mapping = { K: Kv, b: bv, v: Nv, w: Nw }
        Fv2 = F_expression((0,), mapping)
        self.assertAlmostEqual(Fv, Fv2)

        w, b, K = form_data_J.original_coefficients
        v, u = form_data_J.original_arguments
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

    # --- Interaction with replace

    def test_derivative_replace_works_together(self):
        cell = triangle
        V = FiniteElement("CG", cell, 1)

        v = TestFunction(V)
        u = TrialFunction(V)
        f = Coefficient(V)
        g = Coefficient(V)

        M = cos(f)*sin(g)
        F = derivative(M, f, v)
        J = derivative(F, f, u)
        JR = replace(J, { f: g })

        F2 = -sin(f)*v*sin(g)
        J2 = -cos(f)*u*v*sin(g)
        JR2 = -cos(g)*u*v*sin(g)

        self.assertEqualBySampling(F, F2)
        self.assertEqualBySampling(J, J2)
        self.assertEqualBySampling(JR, JR2)

    # --- Scratch space

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
