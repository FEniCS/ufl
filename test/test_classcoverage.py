__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-09-06 -- 2009-02-10"

import ufl
from ufl import *  # noqa: F403, F401
from ufl import (And, Argument, CellDiameter, CellVolume, Circumradius, Coefficient, Constant, FacetArea, FacetNormal,
                 FunctionSpace, Identity, Jacobian, JacobianDeterminant, JacobianInverse, MaxFacetEdgeLength, Mesh,
                 MinFacetEdgeLength, Not, Or, PermutationSymbol, SpatialCoordinate, TensorConstant, VectorConstant,
                 acos, action, as_matrix, as_tensor, as_ufl, as_vector, asin, atan, cell_avg, cofac, conditional, cos,
                 cosh, cross, curl, derivative, det, dev, diff, div, dot, ds, dS, dx, eq, exp, facet_avg, ge, grad, gt,
                 i, inner, inv, j, k, l, le, ln, lt, nabla_div, nabla_grad, ne, outer, rot, sin, sinh, skew, sqrt, sym,
                 tan, tanh, tetrahedron, tr, transpose, triangle, variable)
from ufl.algorithms import *  # noqa: F403, F401
from ufl.classes import *  # noqa: F403, F401
from ufl.classes import (Acos, Asin, Atan, CellCoordinate, Cos, Cosh, Exp, Expr, FacetJacobian,
                         FacetJacobianDeterminant, FacetJacobianInverse, FloatValue, IntValue, Ln, Outer, Sin, Sinh,
                         Sqrt, Tan, Tanh, all_ufl_classes)
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1

has_repr = set()
has_dict = set()


def _test_object(a, shape, free_indices):
    # Check if instances of this type has certain memory consuming members
    if hasattr(a, '_repr'):
        has_repr.add(a.__class__.__name__)
    if hasattr(a, '__dict__'):
        has_dict.add(a.__class__.__name__)

    # Test reproduction via repr string
    r = repr(a)
    e = eval(r, globals())
    assert hash(a) == hash(e)

    # Can't really test str more than that it exists
    str(a)

    # Check that some properties are at least available
    fi = a.ufl_free_indices
    sh = a.ufl_shape

    # Compare with provided properties
    if free_indices is not None:
        free_indices = [i.count() for i in free_indices]
        if len(set(fi) ^ set(free_indices)) != 0:
            print(type(a))
            print(a)
            print(fi)
            print(free_indices)
        assert len(set(fi) ^ set(free_indices)) == 0
    if shape is not None:
        if sh != shape:
            print(("sh:", sh))
            print(("shape:", shape))
        assert sh == shape


def _test_object2(a):
    # Check if instances of this type has certain memory consuming members
    if hasattr(a, '_repr'):
        has_repr.add(a.__class__.__name__)
    if hasattr(a, '__dict__'):
        has_dict.add(a.__class__.__name__)

    # Test reproduction via repr string
    r = repr(a)
    e = eval(r, globals())
    assert hash(a) == hash(e)

    # Can't really test str more than that it exists
    str(a)


def _test_form(a):
    # Test reproduction via repr string
    r = repr(a)
    e = eval(r, globals())
    assert hash(a) == hash(e)

    # Can't really test str more than that it exists
    str(a)


def testExports(self):
    "Verify that ufl.classes exports all Expr subclasses."
    all_expr_classes = []
    for m in list(vars(ufl).values()):
        if isinstance(m, type(ufl)):
            for c in list(vars(m).values()):
                if isinstance(c, type) and issubclass(c, Expr):
                    all_expr_classes.append(c)
    missing_classes = set(c.__name__ for c in all_expr_classes)\
        - set(c.__name__ for c in all_ufl_classes)
    if missing_classes:
        print("The following subclasses of Expr were not exported from ufl.classes:")
        print(("\n".join(sorted(missing_classes))))
    assert missing_classes == set()


def testAll(self):

    Expr.ufl_enable_profiling()

    # --- Elements:
    cell = triangle
    dim = cell.geometric_dimension()

    e0 = FiniteElement("Lagrange", cell, 1, (), identity_pull_back, H1)
    e1 = FiniteElement("Lagrange", cell, 1, (2, ), identity_pull_back, H1)
    e2 = FiniteElement("Lagrange", cell, 1, (2, 2), identity_pull_back, H1)
    e3 = MixedElement([e0, e1, e2])

    e13D = FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1)

    domain = Mesh(FiniteElement("Lagrange", cell, 1, (dim, ), identity_pull_back, H1))
    domain3D = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    e0_space = FunctionSpace(domain, e0)
    e1_space = FunctionSpace(domain, e1)
    e2_space = FunctionSpace(domain, e2)
    e3_space = FunctionSpace(domain, e3)
    e13d_space = FunctionSpace(domain3D, e13D)

    # --- Terminals:

    v13D = Argument(e13d_space, 3)
    f13D = Coefficient(e13d_space)

    v0 = Argument(e0_space, 4)
    v1 = Argument(e1_space, 5)
    v2 = Argument(e2_space, 6)
    v3 = Argument(e3_space, 7)

    _test_object(v0, (), ())
    _test_object(v1, (dim,), ())
    _test_object(v2, (dim, dim), ())
    _test_object(v3, (1 + dim + dim ** 2, ), ())

    f0 = Coefficient(e0_space)
    f1 = Coefficient(e1_space)
    f2 = Coefficient(e2_space)
    f3 = Coefficient(e3_space)

    _test_object(f0, (), ())
    _test_object(f1, (dim,), ())
    _test_object(f2, (dim, dim), ())
    _test_object(f3, (1 + dim + dim ** 2, ), ())

    c = Constant(domain)
    _test_object(c, (), ())

    c = VectorConstant(domain)
    _test_object(c, (dim,), ())

    c = TensorConstant(domain)
    _test_object(c, (dim, dim), ())

    a = FloatValue(1.23)
    _test_object(a, (), ())

    a = IntValue(123)
    _test_object(a, (), ())

    ident = Identity(1)
    _test_object(ident, (1, 1), ())
    ident = Identity(2)
    _test_object(ident, (2, 2), ())
    ident = Identity(3)
    _test_object(ident, (3, 3), ())

    e = PermutationSymbol(2)
    _test_object(e, (2, 2), ())
    e = PermutationSymbol(3)
    _test_object(e, (3, 3, 3), ())

    x = SpatialCoordinate(domain)
    _test_object(x, (dim,), ())
    xi = CellCoordinate(domain)
    _test_object(xi, (dim,), ())

    # g = CellBarycenter(domain)
    # _test_object(g, (dim,), ())
    # g = FacetBarycenter(domain)
    # _test_object(g, (dim,), ())

    g = Jacobian(domain)
    _test_object(g, (dim, dim), ())
    g = JacobianDeterminant(domain)
    _test_object(g, (), ())
    g = JacobianInverse(domain)
    _test_object(g, (dim, dim), ())

    g = FacetJacobian(domain)
    _test_object(g, (dim, dim-1), ())
    g = FacetJacobianDeterminant(domain)
    _test_object(g, (), ())
    g = FacetJacobianInverse(domain)
    _test_object(g, (dim-1, dim), ())

    g = FacetNormal(domain)
    _test_object(g, (dim,), ())
    # g = CellNormal(domain)
    # _test_object(g, (dim,), ())

    g = CellVolume(domain)
    _test_object(g, (), ())
    g = CellDiameter(domain)
    _test_object(g, (), ())
    g = Circumradius(domain)
    _test_object(g, (), ())
    # g = CellSurfaceArea(domain)
    # _test_object(g, (), ())

    g = FacetArea(domain)
    _test_object(g, (), ())
    g = MinFacetEdgeLength(domain)
    _test_object(g, (), ())
    g = MaxFacetEdgeLength(domain)
    _test_object(g, (), ())
    # g = FacetDiameter(domain)
    # _test_object(g, (), ())

    a = variable(v0)
    _test_object(a, (), ())
    a = variable(v1)
    _test_object(a, (dim,), ())
    a = variable(v2)
    _test_object(a, (dim, dim), ())
    a = variable(v3)
    _test_object(a, (1 + dim + dim ** 2, ), ())
    a = variable(f0)
    _test_object(a, (), ())
    a = variable(f1)
    _test_object(a, (dim,), ())
    a = variable(f2)
    _test_object(a, (dim, dim), ())
    a = variable(f3)
    _test_object(a, (1 + dim + dim ** 2, ), ())

    # a = MultiIndex()

    # --- Non-terminals:

    # a = Indexed()
    a = v2[i, j]
    _test_object(a, (), (i, j))
    a = v2[0, k]
    _test_object(a, (), (k,))
    a = v2[l, 1]
    _test_object(a, (), (l,))
    a = f2[i, j]
    _test_object(a, (), (i, j))
    a = f2[0, k]
    _test_object(a, (), (k,))
    a = f2[l, 1]
    _test_object(a, (), (l,))

    ident = Identity(dim)
    a = inv(ident)
    _test_object(a, (dim, dim), ())
    a = inv(v2)
    _test_object(a, (dim, dim), ())
    a = inv(f2)
    _test_object(a, (dim, dim), ())

    for v in (v0, v1, v2, v3):
        for f in (f0, f1, f2, f3):
            a = outer(v, f)
            _test_object(a, None, None)

    for v, f in zip((v0, v1, v2, v3), (f0, f1, f2, f3)):
        a = inner(v, f)
        _test_object(a, None, None)

    for v, f in zip((v1, v2, v3), (f1, f2, f3)):
        a = dot(v, f)
        sh = v.ufl_shape[:-1] + f.ufl_shape[1:]
        _test_object(a, sh, None)

    a = cross(v13D, f13D)
    _test_object(a, (3,), ())

    # a = Sum()
    a = v0 + f0 + v0
    _test_object(a, (), ())
    a = v1 + f1 + v1
    _test_object(a, (dim,), ())
    a = v2 + f2 + v2
    _test_object(a, (dim, dim), ())
    # a = Product()
    a = 3*v0*(2.0*v0)*f0*(v0*3.0)
    _test_object(a, (), ())
    # a = Division()
    a = v0 / 2.0
    _test_object(a, (), ())
    a = v0 / f0
    _test_object(a, (), ())
    a = v0 / (f0 + 7)
    _test_object(a, (), ())
    # a = Power()
    a = f0**3
    _test_object(a, (), ())
    a = (f0*2)**1.23
    _test_object(a, (), ())

    # a = ListTensor()
    a = as_vector([1.0, 2.0*f0, f0**2])
    _test_object(a, (3,), ())
    a = as_matrix([[1.0, 2.0*f0, f0**2],
                   [1.0, 2.0*f0, f0**2]])
    _test_object(a, (2, 3), ())
    a = as_tensor([[[0.00, 0.01, 0.02],
                    [0.10, 0.11, 0.12]],
                   [[1.00, 1.01, 1.02],
                    [1.10, 1.11, 1.12]]])
    _test_object(a, (2, 2, 3), ())

    # a = ComponentTensor()
    a = as_vector(v1[i]*f1[j], i)
    _test_object(a, (dim,), (j,))
    a = as_matrix(v1[i]*f1[j], (j, i))
    _test_object(a, (dim, dim), ())
    a = as_tensor(v1[i]*f1[j], (i, j))
    _test_object(a, (dim, dim), ())
    a = as_tensor(v2[i, j]*f2[j, k], (i, k))
    _test_object(a, (dim, dim), ())

    a = dev(v2)
    _test_object(a, (dim, dim), ())
    a = dev(f2)
    _test_object(a, (dim, dim), ())
    a = dev(f2*f0+v2*3)
    _test_object(a, (dim, dim), ())

    a = sym(v2)
    _test_object(a, (dim, dim), ())
    a = sym(f2)
    _test_object(a, (dim, dim), ())
    a = sym(f2*f0+v2*3)
    _test_object(a, (dim, dim), ())

    a = skew(v2)
    _test_object(a, (dim, dim), ())
    a = skew(f2)
    _test_object(a, (dim, dim), ())
    a = skew(f2*f0+v2*3)
    _test_object(a, (dim, dim), ())

    a = v2.T
    _test_object(a, (dim, dim), ())
    a = f2.T
    _test_object(a, (dim, dim), ())
    a = transpose(f2*f0+v2*3)
    _test_object(a, (dim, dim), ())

    a = det(v2)
    _test_object(a, (), ())
    a = det(f2)
    _test_object(a, (), ())
    a = det(f2*f0+v2*3)
    _test_object(a, (), ())

    a = tr(v2)
    _test_object(a, (), ())
    a = tr(f2)
    _test_object(a, (), ())
    a = tr(f2*f0+v2*3)
    _test_object(a, (), ())

    a = cofac(v2)
    _test_object(a, (dim, dim), ())
    a = cofac(f2)
    _test_object(a, (dim, dim), ())
    a = cofac(f2*f0+v2*3)
    _test_object(a, (dim, dim), ())

    cond1 = le(f0, 1.0)
    cond2 = eq(3.0, f0)
    cond3 = ne(sin(f0), cos(f0))
    cond4 = lt(sin(f0), cos(f0))
    cond5 = ge(sin(f0), cos(f0))
    cond6 = gt(sin(f0), cos(f0))
    cond7 = And(cond1, cond2)
    cond8 = Or(cond1, cond2)
    cond9 = Not(cond8)
    a = conditional(cond1, 1, 2)
    b = conditional(cond2, f0**3, ln(f0))

    _test_object2(cond1)
    _test_object2(cond2)
    _test_object2(cond3)
    _test_object2(cond4)
    _test_object2(cond5)
    _test_object2(cond6)
    _test_object2(cond7)
    _test_object2(cond8)
    _test_object2(cond9)
    _test_object(a, (), ())
    _test_object(b, (), ())

    a = abs(f0)
    _test_object(a, (), ())
    a = sqrt(f0)
    _test_object(a, (), ())
    a = cos(f0)
    _test_object(a, (), ())
    a = sin(f0)
    _test_object(a, (), ())
    a = tan(f0)
    _test_object(a, (), ())
    a = cosh(f0)
    _test_object(a, (), ())
    a = sinh(f0)
    _test_object(a, (), ())
    a = tanh(f0)
    _test_object(a, (), ())
    a = exp(f0)
    _test_object(a, (), ())
    a = ln(f0)
    _test_object(a, (), ())
    a = asin(f0)
    _test_object(a, (), ())
    a = acos(f0)
    _test_object(a, (), ())
    a = atan(f0)
    _test_object(a, (), ())

    one = as_ufl(1)
    a = abs(one)
    _test_object(a, (), ())
    a = Sqrt(one)
    _test_object(a, (), ())
    a = Cos(one)
    _test_object(a, (), ())
    a = Sin(one)
    _test_object(a, (), ())
    a = Tan(one)
    _test_object(a, (), ())
    a = Cosh(one)
    _test_object(a, (), ())
    a = Sinh(one)
    _test_object(a, (), ())
    a = Tanh(one)
    _test_object(a, (), ())
    a = Acos(one)
    _test_object(a, (), ())
    a = Asin(one)
    _test_object(a, (), ())
    a = Atan(one)
    _test_object(a, (), ())
    a = Exp(one)
    _test_object(a, (), ())
    a = Ln(one)
    _test_object(a, (), ())

    # TODO:

    # a = SpatialDerivative()
    a = f0.dx(0)
    _test_object(a, (), ())
    a = f0.dx(i)
    _test_object(a, (), (i,))
    a = f0.dx(i, j, 1)
    _test_object(a, (), (i, j))

    s0 = variable(f0)
    s1 = variable(f1)
    s2 = variable(f2)
    f = dot(s0*s1, s2)
    _test_object(s0, (), ())
    _test_object(s1, (dim,), ())
    _test_object(s2, (dim, dim), ())
    _test_object(f, (dim,), ())

    a = diff(f, s0)
    _test_object(a, (dim,), ())
    a = diff(f, s1)
    _test_object(a, (dim, dim,), ())
    a = diff(f, s2)
    _test_object(a, (dim, dim, dim), ())

    a = div(v1)
    _test_object(a, (), ())
    a = div(f1)
    _test_object(a, (), ())
    a = div(v2)
    _test_object(a, (dim,), ())
    a = div(f2)
    _test_object(a, (dim,), ())
    a = div(Outer(f1, f1))
    _test_object(a, (dim,), ())

    a = grad(v0)
    _test_object(a, (dim,), ())
    a = grad(f0)
    _test_object(a, (dim,), ())
    a = grad(v1)
    _test_object(a, (dim, dim), ())
    a = grad(f1)
    _test_object(a, (dim, dim), ())
    a = grad(f0*v0)
    _test_object(a, (dim,), ())
    a = grad(f0*v1)
    _test_object(a, (dim, dim), ())

    a = nabla_div(v1)
    _test_object(a, (), ())
    a = nabla_div(f1)
    _test_object(a, (), ())
    a = nabla_div(v2)
    _test_object(a, (dim,), ())
    a = nabla_div(f2)
    _test_object(a, (dim,), ())
    a = nabla_div(Outer(f1, f1))
    _test_object(a, (dim,), ())

    a = nabla_grad(v0)
    _test_object(a, (dim,), ())
    a = nabla_grad(f0)
    _test_object(a, (dim,), ())
    a = nabla_grad(v1)
    _test_object(a, (dim, dim), ())
    a = nabla_grad(f1)
    _test_object(a, (dim, dim), ())
    a = nabla_grad(f0*v0)
    _test_object(a, (dim,), ())
    a = nabla_grad(f0*v1)
    _test_object(a, (dim, dim), ())

    a = curl(v13D)
    _test_object(a, (3,), ())
    a = curl(f13D)
    _test_object(a, (3,), ())
    a = rot(v1)
    _test_object(a, (), ())
    a = rot(f1)
    _test_object(a, (), ())

    # a = PositiveRestricted(v0)
    # _test_object(a, (), ())
    a = v0('+')
    _test_object(a, (), ())
    a = v0('+')*f0
    _test_object(a, (), ())

    # a = NegativeRestricted(v0)
    # _test_object(a, (), ())
    a = v0('-')
    _test_object(a, (), ())
    a = v0('-') + f0
    _test_object(a, (), ())

    a = cell_avg(v0)
    _test_object(a, (), ())
    a = facet_avg(v0)
    _test_object(a, (), ())
    a = cell_avg(v1)
    _test_object(a, (dim,), ())
    a = facet_avg(v1)
    _test_object(a, (dim,), ())
    a = cell_avg(v1)[i]
    _test_object(a, (), (i,))
    a = facet_avg(v1)[i]
    _test_object(a, (), (i,))

    # --- Integrals:

    a = v0*dx
    _test_form(a)
    a = v0*dx(0)
    _test_form(a)
    a = v0*dx(1)
    _test_form(a)
    a = v0*ds
    _test_form(a)
    a = v0*ds(0)
    _test_form(a)
    a = v0*ds(1)
    _test_form(a)
    a = v0*dS
    _test_form(a)
    a = v0*dS(0)
    _test_form(a)
    a = v0*dS(1)
    _test_form(a)

    a = v0*dot(v1, f1)*dx
    _test_form(a)
    a = v0*dot(v1, f1)*dx(0)
    _test_form(a)
    a = v0*dot(v1, f1)*dx(1)
    _test_form(a)
    a = v0*dot(v1, f1)*ds
    _test_form(a)
    a = v0*dot(v1, f1)*ds(0)
    _test_form(a)
    a = v0*dot(v1, f1)*ds(1)
    _test_form(a)
    a = v0*dot(v1, f1)*dS
    _test_form(a)
    a = v0*dot(v1, f1)*dS(0)
    _test_form(a)
    a = v0*dot(v1, f1)*dS(1)
    _test_form(a)

    # --- Form transformations:

    a = f0*v0*dx + f0*v0*dot(f1, v1)*dx
    # b = lhs(a) # TODO
    # c = rhs(a) # TODO
    d = derivative(a, f1, v1)
    f = action(d)
    # e = action(b)

    # --- Check which classes have been created
    ic, dc = Expr.ufl_disable_profiling()

    constructed = set()
    unused = set(Expr._ufl_all_classes_)
    for cls in Expr._ufl_all_classes_:
        tc = cls._ufl_typecode_
        if ic[tc]:
            constructed.add(cls)
        if cls._ufl_is_abstract_:
            unused.remove(cls)

    if unused:
        print()
        print("The following classes were never instantiated in class coverage test:")
        print(("\n".join(sorted(map(str, unused)))))
        print()

    # --- Check which classes had certain member variables
    if has_repr:
        print()
        print("The following classes contain a _repr member:")
        print(("\n".join(sorted(map(str, has_repr)))))
        print()

    if has_dict:
        print()
        print("The following classes contain a __dict__ member:")
        print(("\n".join(sorted(map(str, has_dict)))))
        print()

    # TODO: Add tests for bessel functions:
    #   BesselI
    #   BesselJ
    #   BesselK
    #   BesselY
    #   Erf
    # TODO: Add tests for:
    #   Label
