__authors__ = "Martin Sandve Alnæs"
__date__ = "2009-02-13 -- 2009-02-13"

import math

from ufl import (Argument, Coefficient, FiniteElement, FunctionSpace, Identity, Mesh, SpatialCoordinate, VectorElement,
                 as_matrix, as_vector, cos, cross, det, dev, dot, exp, i, indices, inner, j, ln, outer, sin, skew, sqrt,
                 sym, tan, tetrahedron, tr, triangle)
from ufl.constantvalue import as_ufl
from ufl.finiteelement import FiniteElement
from ufl.sobolevspace import H1


def testScalars():
    s = as_ufl(123)
    e = s((5, 7))
    v = 123
    assert e == v


def testZero():
    s = as_ufl(0)
    e = s((5, 7))
    v = 0
    assert e == v


def testIdentity():
    cell = triangle
    ident = Identity(cell.geometric_dimension())

    s = 123 * ident[0, 0]
    e = s((5, 7))
    v = 123
    assert e == v

    s = 123 * ident[1, 0]
    e = s((5, 7))
    v = 0
    assert e == v


def testCoords():
    cell = triangle
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    x = SpatialCoordinate(domain)
    s = x[0] + x[1]
    e = s((5, 7))
    v = 5 + 7
    assert e == v


def testFunction1():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    s = 3 * f
    e = s((5, 7), {f: 123})
    v = 3 * 123
    assert e == v


def testFunction2():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    space = FunctionSpace(domain, element)
    f = Coefficient(space)

    def g(x):
        return x[0]
    s = 3 * f
    e = s((5, 7), {f: g})
    v = 3 * 5
    assert e == v


def testArgument2():
    cell = triangle
    element = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    space = FunctionSpace(domain, element)
    f = Argument(space, 2)

    def g(x):
        return x[0]
    s = 3 * f
    e = s((5, 7), {f: g})
    v = 3 * 5
    assert e == v


def testAlgebra():
    cell = triangle
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    x = SpatialCoordinate(domain)
    s = 3 * (x[0] + x[1]) - 7 + x[0] ** (x[1] / 2)
    e = s((5, 7))
    v = 3 * (5. + 7.) - 7 + 5. ** (7. / 2)
    assert e == v


def testIndexSum():
    cell = triangle
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    x = SpatialCoordinate(domain)
    i, = indices(1)
    s = x[i] * x[i]
    e = s((5, 7))
    v = 5 ** 2 + 7 ** 2
    assert e == v


def testIndexSum2():
    cell = triangle
    domain = Mesh(FiniteElement("Lagrange", cell, 1, (d, ), (d, ), "identity", H1))
    x = SpatialCoordinate(domain)
    ident = Identity(cell.geometric_dimension())
    i, j = indices(2)
    s = (x[i] * x[j]) * ident[i, j]
    e = s((5, 7))
    # v = sum_i sum_j x_i x_j delta_ij = x_0 x_0 + x_1 x_1
    v = 5 ** 2 + 7 ** 2
    assert e == v


def testMathFunctions():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)[0]

    s = sin(x)
    e = s((5, 7))
    v = math.sin(5)
    assert e == v

    s = cos(x)
    e = s((5, 7))
    v = math.cos(5)
    assert e == v

    s = tan(x)
    e = s((5, 7))
    v = math.tan(5)
    assert e == v

    s = ln(x)
    e = s((5, 7))
    v = math.log(5)
    assert e == v

    s = exp(x)
    e = s((5, 7))
    v = math.exp(5)
    assert e == v

    s = sqrt(x)
    e = s((5, 7))
    v = math.sqrt(5)
    assert e == v


def testListTensor():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x, y = SpatialCoordinate(domain)

    m = as_matrix([[x, y], [-y, -x]])

    s = m[0, 0] + m[1, 0] + m[0, 1] + m[1, 1]
    e = s((5, 7))
    v = 0
    assert e == v

    s = m[0, 0] * m[1, 0] * m[0, 1] * m[1, 1]
    e = s((5, 7))
    v = 5 ** 2 * 7 ** 2
    assert e == v


def testComponentTensor1():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    m = as_vector(x[i], i)

    s = m[0] * m[1]
    e = s((5, 7))
    v = 5 * 7
    assert e == v


def testComponentTensor2():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xx = outer(x, x)

    m = as_matrix(xx[i, j], (i, j))

    s = m[0, 0] + m[1, 0] + m[0, 1] + m[1, 1]
    e = s((5, 7))
    v = 5 * 5 + 5 * 7 + 5 * 7 + 7 * 7
    assert e == v


def testComponentTensor3():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xx = outer(x, x)

    m = as_matrix(xx[i, j], (i, j))

    s = m[0, 0] * m[1, 0] * m[0, 1] * m[1, 1]
    e = s((5, 7))
    v = 5 * 5 * 5 * 7 * 5 * 7 * 7 * 7
    assert e == v


def testCoefficient():
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, V)
    f = Coefficient(space)
    e = f ** 2

    def eval_f(x):
        return x[0] * x[1] ** 2
    assert e((3, 7), {f: eval_f}) == (3 * 7 ** 2) ** 2


def testCoefficientDerivative():
    V = FiniteElement("Lagrange", triangle, 1, (), (), "identity", H1)
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    space = FunctionSpace(domain, V)
    f = Coefficient(space)
    e = f.dx(0) ** 2 + f.dx(1) ** 2

    def eval_f(x, derivatives):
        if not derivatives:
            return eval_f.c * x[0] * x[1] ** 2
        # assume only first order derivative
        d, = derivatives
        if d == 0:
            return eval_f.c * x[1] ** 2
        if d == 1:
            return eval_f.c * x[0] * 2 * x[1]
    # shows how to attach data to eval_f
    eval_f.c = 5

    assert e((3, 7), {f: eval_f}) == (5 * 7 ** 2) ** 2 + (5 * 3 * 2 * 7) ** 2


def test_dot():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    s = dot(x, 2 * x)
    e = s((5, 7))
    v = 2 * (5 * 5 + 7 * 7)
    assert e == v


def test_inner():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xx = as_matrix(((2 * x[0], 3 * x[0]), (2 * x[1], 3 * x[1])))
    s = inner(xx, 2 * xx)
    e = s((5, 7))
    v = 2 * ((5 * 2) ** 2 + (5 * 3) ** 2 + (7 * 2) ** 2 + (7 * 3) ** 2)
    assert e == v


def test_outer():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xx = outer(outer(x, x), as_vector((2, 3)))
    s = inner(xx, 2 * xx)
    e = s((5, 7))
    v = 2 * (5 ** 2 + 7 ** 2) ** 2 * (2 ** 2 + 3 ** 2)
    assert e == v


def test_cross():
    domain = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), (3, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xv = (3, 5, 7)

    # Test cross product of triplets of orthogonal
    # vectors, where |a x b| = |a| |b|
    ts = [
        [as_vector((x[0], 0, 0)),
            as_vector((0, x[1], 0)),
            as_vector((0, 0, x[2]))],
        [as_vector((x[0], x[1], 0)),
            as_vector((x[1], -x[0], 0)),
            as_vector((0, 0, x[2]))],
        [as_vector((0, x[0], x[1])),
            as_vector((0, x[1], -x[0])),
            as_vector((x[2], 0, 0))],
        [as_vector((x[0], 0, x[1])),
            as_vector((x[1], 0, -x[0])),
            as_vector((0, x[2], 0))],
    ]
    for t in ts:
        for a in range(3):
            for b in range(3):
                cab = cross(t[a], t[b])
                dab = dot(cab, cab)
                eab = dab(xv)
                tna = dot(t[a], t[a])(xv)
                tnb = dot(t[b], t[b])(xv)
                vab = tna * tnb if a != b else 0
                assert eab == vab


def xtest_dev():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xv = (5, 7)
    xx = outer(x, x)
    s1 = dev(2 * xx)
    s2 = 2 * (xx - xx.T)  # FIXME
    e = inner(s1, s1)(xv)
    v = inner(s2, s2)(xv)
    assert e == v


def test_skew():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xv = (5, 7)
    xx = outer(x, x)
    s1 = skew(2 * xx)
    s2 = (xx - xx.T)
    e = inner(s1, s1)(xv)
    v = inner(s2, s2)(xv)
    assert e == v


def test_sym():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xv = (5, 7)
    xx = outer(x, x)
    s1 = sym(2 * xx)
    s2 = (xx + xx.T)
    e = inner(s1, s1)(xv)
    v = inner(s2, s2)(xv)
    assert e == v


def test_tr():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xv = (5, 7)
    xx = outer(x, x)
    s = tr(2 * xx)
    e = s(xv)
    v = 2 * sum(xv[i] ** 2 for i in (0, 1))
    assert e == v


def test_det2D():
    domain = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), "identity", H1))
    x = SpatialCoordinate(domain)
    xv = (5, 7)
    a, b = 6.5, -4
    xx = as_matrix(((x[0], x[1]), (a, b)))
    s = det(2 * xx)
    e = s(xv)
    v = 2 ** 2 * (5 * b - 7 * a)
    assert e == v


def xtest_det3D():  # FIXME
    x = SpatialCoordinate(tetrahedron)
    xv = (5, 7, 9)
    a, b, c = 6.5, -4, 3
    d, e, f = 2, 3, 4
    xx = as_matrix(((x[0], x[1], x[2]),
                    (a, b, c),
                    (d, e, f)))
    s = det(2 * xx)
    e = s(xv)
    v = 2 ** 3 * \
        (xv[0] * (b * f - e * c) - xv[1] *
         (a * f - c * d) + xv[2] * (a * e - b * d))
    assert e == v


def test_cofac():
    pass  # TODO


def test_inv():
    pass  # TODO
