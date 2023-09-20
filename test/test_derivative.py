__authors__ = "Martin Sandve Alnæs"
__date__ = "2009-02-17 -- 2009-02-17"

from itertools import chain

from ufl import (CellDiameter, CellVolume, Circumradius, Coefficient, Constant, FacetArea, FacetNormal, FunctionSpace,
                 Identity, Index, Jacobian, JacobianInverse, Mesh, SpatialCoordinate, TestFunction, TrialFunction, acos,
                 as_matrix, as_tensor, as_vector, asin, atan, conditional, cos, derivative, diff, dot, dx, exp, i,
                 indices, inner, interval, j, k, ln, lt, nabla_grad, outer, quadrilateral, replace, sign, sin, split,
                 sqrt, tan, tetrahedron, triangle, variable, zero)
from ufl.algorithms import compute_form_data, expand_indices, strip_variables
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.classes import Indexed, MultiIndex, ReferenceGrad
from ufl.constantvalue import as_ufl
from ufl.domain import extract_unique_domain
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.sobolevspace import H1, L2


def assertEqualBySampling(actual, expected):
    ad = compute_form_data(actual*dx)
    a = ad.preprocessed_form.integrals_by_type("cell")[0].integrand()
    bd = compute_form_data(expected*dx)
    b = bd.preprocessed_form.integrals_by_type("cell")[0].integrand()

    assert ([ad.function_replace_map[ac] for ac in ad.reduced_coefficients]
            == [bd.function_replace_map[bc] for bc in bd.reduced_coefficients])

    n = ad.num_coefficients

    def make_value(c):
        if isinstance(c, Coefficient):
            z = 0.3
            m = c.count()
        else:
            z = 0.7
            m = c.number()
        if c.ufl_shape == ():
            return z * (0.1 + 0.9 * m / n)
        elif len(c.ufl_shape) == 1:
            return tuple((z * (j + 0.1 + 0.9 * m / n) for j in range(c.ufl_shape[0])))
        else:
            raise NotImplementedError("Tensor valued expressions not supported here.")

    amapping = dict((c, make_value(c)) for c in chain(ad.original_form.coefficients(), ad.original_form.arguments()))
    bmapping = dict((c, make_value(c)) for c in chain(bd.original_form.coefficients(), bd.original_form.arguments()))
    acell = extract_unique_domain(actual).ufl_cell()
    bcell = extract_unique_domain(expected).ufl_cell()
    assert acell == bcell
    if acell.geometric_dimension() == 1:
        x = (0.3,)
    elif acell.geometric_dimension() == 2:
        x = (0.3, 0.4)
    elif acell.geometric_dimension() == 3:
        x = (0.3, 0.4, 0.5)
    av = a(x, amapping)
    bv = b(x, bmapping)

    if not av == bv:
        print("Tried to sample expressions to compare but failed:")
        print()
        print((str(a)))
        print(av)
        print()
        print((str(b)))
        print(bv)
        print()

    assert av == bv


def _test(self, f, df):
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)

    v = TestFunction(space)
    u = TrialFunction(space)
    w = Coefficient(space)
    xv = (0.3, 0.7)
    uv = 7.0
    vv = 13.0
    wv = 11.0

    x = xv
    mapping = {v: vv, u: uv, w: wv}

    dfv1 = derivative(f(w), w, v)
    dfv2 = df(w, v)
    dfv1 = dfv1(x, mapping)
    dfv2 = dfv2(x, mapping)
    assert dfv1 == dfv2

    dfv1 = derivative(f(7*w), w, v)
    dfv2 = 7*df(7*w, v)
    dfv1 = dfv1(x, mapping)
    dfv2 = dfv2(x, mapping)
    assert dfv1 == dfv2

# --- Literals


def testScalarLiteral(self):
    def f(w): return as_ufl(1)

    def df(w, v): return zero()
    _test(self, f, df)


def testIdentityLiteral(self):
    def f(w): return Identity(2)[i, i]

    def df(w, v): return zero()
    _test(self, f, df)

# --- Form arguments


def testCoefficient(self):
    def f(w): return w

    def df(w, v): return v
    _test(self, f, df)


def testArgument(self):
    def f(w):
        return TestFunction(FunctionSpace(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)),
                                          FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)))

    def df(w, v):
        return zero()
    _test(self, f, df)

# --- Geometry


def testSpatialCoordinate(self):
    def f(w): return SpatialCoordinate(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)))[0]

    def df(w, v): return zero()
    _test(self, f, df)


def testFacetNormal(self):
    def f(w): return FacetNormal(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)))[0]

    def df(w, v): return zero()
    _test(self, f, df)

# def testCellSurfaceArea(self):
#    def f(w):     return CellSurfaceArea(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)))
#    def df(w, v): return zero()
#    _test(self, f, df)


def testFacetArea(self):
    def f(w): return FacetArea(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)))

    def df(w, v): return zero()
    _test(self, f, df)


def testCellDiameter(self):
    def f(w): return CellDiameter(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)))

    def df(w, v): return zero()
    _test(self, f, df)


def testCircumradius(self):
    def f(w): return Circumradius(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)))

    def df(w, v): return zero()
    _test(self, f, df)


def testCellVolume(self):
    def f(w): return CellVolume(Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)))

    def df(w, v): return zero()
    _test(self, f, df)

# --- Basic operators


def testSum(self):
    def f(w): return w + 1

    def df(w, v): return v
    _test(self, f, df)


def testProduct(self):
    def f(w): return 3*w

    def df(w, v): return 3*v
    _test(self, f, df)


def testPower(self):
    def f(w): return w**3

    def df(w, v): return 3*w**2*v
    _test(self, f, df)


def testDivision(self):
    def f(w): return w / 3.0

    def df(w, v): return v / 3.0
    _test(self, f, df)


def testDivision2(self):
    def f(w): return 3.0 / w

    def df(w, v): return -3.0 * v / w**2
    _test(self, f, df)


def testExp(self):
    def f(w): return exp(w)

    def df(w, v): return v*exp(w)
    _test(self, f, df)


def testLn(self):
    def f(w): return ln(w)

    def df(w, v): return v / w
    _test(self, f, df)


def testCos(self):
    def f(w): return cos(w)

    def df(w, v): return -v*sin(w)
    _test(self, f, df)


def testSin(self):
    def f(w): return sin(w)

    def df(w, v): return v*cos(w)
    _test(self, f, df)


def testTan(self):
    def f(w): return tan(w)

    def df(w, v): return v*2.0/(cos(2.0*w) + 1.0)
    _test(self, f, df)


def testAcos(self):
    def f(w): return acos(w/1000)

    def df(w, v): return -(v/1000)/sqrt(1.0 - (w/1000)**2)
    _test(self, f, df)


def testAsin(self):
    def f(w): return asin(w/1000)

    def df(w, v): return (v/1000)/sqrt(1.0 - (w/1000)**2)
    _test(self, f, df)


def testAtan(self):
    def f(w): return atan(w)

    def df(w, v): return v/(1.0 + w**2)
    _test(self, f, df)

# FIXME: Add the new erf and bessel_*

# --- Abs and conditionals


def testAbs(self):
    def f(w): return abs(w)

    def df(w, v): return sign(w)*v
    _test(self, f, df)


def testConditional(self):  # This will fail without bugfix in derivative
    def cond(w): return lt(w, 1.0)

    def f(w): return conditional(cond(w), 2*w, 3*w)

    def df(w, v): return (conditional(cond(w), 1, 0) * 2*v +
                          conditional(cond(w), 0, 1) * 3*v)
    _test(self, f, df)

# --- Tensor algebra basics


def testIndexSum(self):
    def f(w):
        # 3*w + 4*w**2 + 5*w**3
        a = as_vector((w, w**2, w**3))
        b = as_vector((3, 4, 5))
        i, = indices(1)
        return a[i]*b[i]

    def df(w, v): return 3*v + 4*2*w*v + 5*3*w**2*v
    _test(self, f, df)


def testListTensor(self):
    v = variable(as_ufl(42))
    f = as_tensor((
        ((0, 0), (0, 0)),
        ((v, 2*v), (0, 0)),
        ((v**2, 1), (2, v/2)),
    ))
    assert f.ufl_shape == (3, 2, 2)
    g = as_tensor((
        ((0, 0), (0, 0)),
        ((1, 2), (0, 0)),
        ((84, 0), (0, 0.5)),
    ))
    assert g.ufl_shape == (3, 2, 2)
    dfv = diff(f, v)
    x = None
    for a in range(3):
        for b in range(2):
            for c in range(2):
                self.assertEqual(dfv[a, b, c](x), g[a, b, c](x))

# --- Coefficient and argument input configurations


def test_single_scalar_coefficient_derivative(self):
    cell = triangle
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, V)
    u = Coefficient(space)
    v = TestFunction(space)
    a = 3*u**2
    b = derivative(a, u, v)
    self.assertEqualAfterPreprocessing(b, 3*(u*(2*v)))


def test_single_vector_coefficient_derivative(self):
    cell = triangle
    V = FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, V)
    u = Coefficient(space)
    v = TestFunction(space)
    a = 3*dot(u, u)
    actual = derivative(a, u, v)
    expected = 3*(2*(u[i]*v[i]))
    assertEqualBySampling(actual, expected)


def test_multiple_coefficient_derivative(self):
    cell = triangle
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    W = FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1)
    M = MixedElement([V, W])
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    w_space = FunctionSpace(domain, W)
    m_space = FunctionSpace(domain, M)
    uv = Coefficient(v_space)
    uw = Coefficient(w_space)
    v = TestFunction(m_space)
    vv, vw = split(v)

    a = sin(uv)*dot(uw, uw)

    actual = derivative(a, (uv, uw), split(v))
    expected = cos(uv)*vv * (uw[i]*uw[i]) + (uw[j]*vw[j])*2 * sin(uv)
    assertEqualBySampling(actual, expected)

    actual = derivative(a, (uv, uw), v)
    expected = cos(uv)*vv * (uw[i]*uw[i]) + (uw[j]*vw[j])*2 * sin(uv)
    assertEqualBySampling(actual, expected)


def test_indexed_coefficient_derivative(self):
    cell = triangle
    ident = Identity(cell.geometric_dimension())
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    W = FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    w_space = FunctionSpace(domain, W)
    u = Coefficient(w_space)
    v = TestFunction(v_space)

    w = dot(u, nabla_grad(u))
    # a = dot(w, w)
    a = (u[i]*u[k].dx(i)) * w[k]

    actual = derivative(a, u[0], v)

    dw = v*u[k].dx(0) + u[i]*ident[0, k]*v.dx(i)
    expected = 2 * w[k] * dw

    assertEqualBySampling(actual, expected)


def test_multiple_indexed_coefficient_derivative(self):
    cell = tetrahedron
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    V2 = MixedElement([V, V])
    W = FiniteElement("Lagrange", cell, 1, (3, ), (3, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (3, ), (3, ), "identity", H1))
    v2_space = FunctionSpace(domain, V2)
    w_space = FunctionSpace(domain, W)
    u = Coefficient(w_space)
    w = Coefficient(w_space)
    v = TestFunction(v2_space)
    vu, vw = split(v)

    actual = derivative(cos(u[i]*w[i]), (u[2], w[1]), (vu, vw))
    expected = -sin(u[i]*w[i])*(vu*w[2] + u[1]*vw)

    assertEqualBySampling(actual, expected)


def test_segregated_derivative_of_convection(self):
    cell = tetrahedron
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    W = FiniteElement("Lagrange", cell, 1, (3, ), (3, ), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", cell, 1, (3, ), (3, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    w_space = FunctionSpace(domain, W)

    u = Coefficient(w_space)
    v = Coefficient(w_space)
    du = TrialFunction(v_space)
    dv = TestFunction(v_space)

    L = dot(dot(u, nabla_grad(u)), v)

    Lv = {}
    Lvu = {}
    for a in range(cell.geometric_dimension()):
        Lv[a] = derivative(L, v[a], dv)
        for b in range(cell.geometric_dimension()):
            Lvu[a, b] = derivative(Lv[a], u[b], du)

    for a in range(cell.geometric_dimension()):
        for b in range(cell.geometric_dimension()):
            form = Lvu[a, b]*dx
            fd = compute_form_data(form)
            pf = fd.preprocessed_form
            expand_indices(pf)

    k = Index()
    for a in range(cell.geometric_dimension()):
        for b in range(cell.geometric_dimension()):
            actual = Lvu[a, b]
            expected = du*u[a].dx(b)*dv + u[k]*du.dx(k)*dv
            assertEqualBySampling(actual, expected)

# --- User provided derivatives of coefficients


def test_coefficient_derivatives(self):
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, V)

    dv = TestFunction(space)

    f = Coefficient(space, count=0)
    g = Coefficient(space, count=1)
    df = Coefficient(space, count=2)
    dg = Coefficient(space, count=3)
    u = Coefficient(space, count=4)
    cd = {f: df, g: dg}

    integrand = inner(f, g)
    expected = (df*dv)*g + f*(dg*dv)

    F = integrand*dx
    J = derivative(F, u, dv, cd)
    fd = compute_form_data(J)
    actual = fd.preprocessed_form.integrals()[0].integrand()
    assert (actual*dx).signature() == (expected*dx).signature()
    self.assertEqual(replace(actual, fd.function_replace_map), expected)


def test_vector_coefficient_scalar_derivatives(self):
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    VV = FiniteElement("vector Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    vv_space = FunctionSpace(domain, VV)

    dv = TestFunction(v_space)

    df = Coefficient(vv_space, count=0)
    g = Coefficient(vv_space, count=1)
    f = Coefficient(vv_space, count=2)
    u = Coefficient(v_space, count=3)
    cd = {f: df}

    integrand = inner(f, g)

    i0, i1, i2, i3, i4 = [Index(count=c) for c in range(5)]
    expected = as_tensor(df[i1]*dv, (i1,))[i0]*g[i0]

    F = integrand*dx
    J = derivative(F, u, dv, cd)
    fd = compute_form_data(J)
    actual = fd.preprocessed_form.integrals()[0].integrand()
    assert (actual*dx).signature() == (expected*dx).signature()


def test_vector_coefficient_derivatives(self):
    V = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    VV = FiniteElement("Lagrange", triangle, 1, (2, 2), (2, 2), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    vv_space = FunctionSpace(domain, VV)

    dv = TestFunction(v_space)

    df = Coefficient(vv_space, count=0)
    g = Coefficient(v_space, count=1)
    f = Coefficient(v_space, count=2)
    u = Coefficient(v_space, count=3)
    cd = {f: df}

    integrand = inner(f, g)

    i0, i1, i2, i3, i4 = [Index(count=c) for c in range(5)]
    expected = as_tensor(df[i2, i1]*dv[i1], (i2,))[i0]*g[i0]

    F = integrand*dx
    J = derivative(F, u, dv, cd)
    fd = compute_form_data(J)
    actual = fd.preprocessed_form.integrals()[0].integrand()
    assert (actual*dx).signature() == (expected*dx).signature()
    # self.assertEqual(replace(actual, fd.function_replace_map), expected)


def test_vector_coefficient_derivatives_of_product(self):
    V = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    VV = FiniteElement("Lagrange", triangle, 1, (2, 2), (2, 2), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    v_space = FunctionSpace(domain, V)
    vv_space = FunctionSpace(domain, VV)

    dv = TestFunction(v_space)

    df = Coefficient(vv_space, count=0)
    g = Coefficient(v_space, count=1)
    dg = Coefficient(vv_space, count=2)
    f = Coefficient(v_space, count=3)
    u = Coefficient(v_space, count=4)
    cd = {f: df, g: dg}

    integrand = f[i]*g[i]

    i0, i1, i2, i3, i4 = [Index(count=c) for c in range(5)]
    expected = as_tensor(df[i2, i1]*dv[i1], (i2,))[i0]*g[i0] +\
        f[i0]*as_tensor(dg[i4, i3]*dv[i3], (i4,))[i0]

    F = integrand*dx
    J = derivative(F, u, dv, cd)
    fd = compute_form_data(J)
    actual = fd.preprocessed_form.integrals()[0].integrand()

    # Tricky case! These are equal in representation except
    # that the outermost sum/indexsum are swapped.
    # Sampling the expressions instead of comparing representations.
    x = (0, 0)
    funcs = {dv: (13, 14), f: (1, 2), g: (3, 4), df: ((5, 6), (7, 8)), dg: ((9, 10), (11, 12))}
    self.assertEqual(replace(actual, fd.function_replace_map)(x, funcs), expected(x, funcs))

    # TODO: Add tests covering more cases, in particular mixed stuff

# --- Some actual forms


def testHyperElasticity(self):
    cell = interval
    element = FiniteElement("Lagrange", cell, 2, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (1, ), (1, ), "identity", H1))
    space = FunctionSpace(domain, element)
    w = Coefficient(space)
    v = TestFunction(space)
    u = TrialFunction(space)
    b = Constant(domain)
    K = Constant(domain)

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

    f_expression = strip_variables(f.integrals_by_type("cell")[0].integrand())
    F_expression = strip_variables(F.integrals_by_type("cell")[0].integrand())
    J_expression = strip_variables(J.integrals_by_type("cell")[0].integrand())

    # classes = set(c.__class__ for c in post_traversal(f_expression))

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

    mapping = {K: Kv, b: bv, w: Nw}
    fv2 = f_expression((0,), mapping)
    self.assertAlmostEqual(fv, fv2)

    v, = form_data_F.original_form.arguments()
    mapping = {K: Kv, b: bv, v: Nv, w: Nw}
    Fv2 = F_expression((0,), mapping)
    self.assertAlmostEqual(Fv, Fv2)

    v, u = form_data_J.original_form.arguments()
    mapping = {K: Kv, b: bv, v: Nv, u: Nu, w: Nw}
    Jv2 = J_expression((0,), mapping)
    self.assertAlmostEqual(Jv, Jv2)


def test_mass_derived_from_functional(self):
    cell = triangle
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, V)

    v = TestFunction(space)
    u = TrialFunction(space)
    w = Coefficient(space)

    f = (w**2/2)*dx
    L = w*v*dx
    a = u*v*dx  # noqa: F841
    F = derivative(f, w, v)
    J1 = derivative(L, w, u)  # noqa: F841
    J2 = derivative(F, w, u)  # noqa: F841
    # TODO: assert something

# --- Interaction with replace


def test_derivative_replace_works_together(self):
    cell = triangle
    V = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)

    domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, V)

    v = TestFunction(space)
    u = TrialFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)

    M = cos(f)*sin(g)
    F = derivative(M, f, v)
    J = derivative(F, f, u)
    JR = replace(J, {f: g})

    F2 = -sin(f)*v*sin(g)
    J2 = -cos(f)*u*v*sin(g)
    JR2 = -cos(g)*u*v*sin(g)

    assertEqualBySampling(F, F2)
    assertEqualBySampling(J, J2)
    assertEqualBySampling(JR, JR2)


def test_index_simplification_handles_repeated_indices(self):
    mesh = Mesh(FiniteElement("Lagrange", quadrilateral, 1, (2, ), (2, ), "identity", H1))
    V = FunctionSpace(mesh, FiniteElement("DQ", quadrilateral, 0, (2, 2), (2, 2), "identity", L2))
    K = JacobianInverse(mesh)
    G = outer(Identity(2), Identity(2))
    i, j, k, l, m, n = indices(6)
    A = as_tensor(K[m, i] * K[n, j] * G[i, j, k, l], (m, n, k, l))
    i, j = indices(2)
    # Can't use A[i, i, j, j] because UFL automagically index-sums
    # repeated indices in the __getitem__ call.
    Adiag = Indexed(A, MultiIndex((i, i, j, j)))
    A = as_tensor(Adiag, (i, j))
    v = TestFunction(V)
    f = inner(A, v)*dx
    fd = compute_form_data(f, do_apply_geometry_lowering=True)
    integral, = fd.preprocessed_form.integrals()
    assert integral.integrand().ufl_free_indices == ()


def test_index_simplification_reference_grad(self):
    mesh = Mesh(FiniteElement("Lagrange", quadrilateral, 1, (2, ), (2, ), "identity", H1))
    i, = indices(1)
    A = as_tensor(Indexed(Jacobian(mesh), MultiIndex((i, i))), (i,))
    expr = apply_derivatives(apply_geometry_lowering(
        apply_algebra_lowering(A[0])))
    assert expr == ReferenceGrad(SpatialCoordinate(mesh))[0, 0]
    assert expr.ufl_free_indices == ()
    assert expr.ufl_shape == ()


# --- Scratch space

def test_foobar(self):
    element = FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)

    du = TrialFunction(space)

    U = Coefficient(space)

    def planarGrad(u):
        return as_matrix([[u[0].dx(0), 0, u[0].dx(1)],
                          [0, 0, 0],
                          [u[1].dx(0), 0, u[1].dx(1)]])

    def epsilon(u):
        return 0.5*(planarGrad(u)+planarGrad(u).T)

    def NS_a(u, v):
        return inner(epsilon(u), epsilon(v))

    L = NS_a(U, v)*dx
    a = derivative(L, U, du)  # noqa: F841
    # TODO: assert something
