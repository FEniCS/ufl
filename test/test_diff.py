__authors__ = "Martin Sandve Alnæs"
__date__ = "2009-02-17 -- 2014-10-14"

import math

import pytest

from ufl import *
from ufl.algorithms import expand_derivatives
from ufl.constantvalue import as_ufl


def get_variables():
    xv = None
    vv = 5.0
    return (xv, variable(vv))


@pytest.fixture
def v():
    xv, vv = get_variables()
    return vv


def _test(f, df):
    x, v = get_variables()

    dfv1 = diff(f(v), v)
    dfv2 = df(v)
    dfv1 = dfv1(x)
    dfv2 = dfv2(x)
    assert round(dfv1 - dfv2, 7) == 0

    dfv1 = diff(f(7 * v), v)
    dfv2 = 7 * df(7 * v)
    dfv1 = dfv1(x)
    dfv2 = dfv2(x)
    assert round(dfv1 - dfv2, 7) == 0


def testVariable(v):
    def f(v):
        return v

    def df(v):
        return as_ufl(1)
    _test(f, df)


def testSum(v):
    def f(v):
        return v + 1

    def df(v):
        return as_ufl(1)
    _test(f, df)


def testProduct(v):
    def f(v):
        return 3 * v

    def df(v):
        return as_ufl(3)
    _test(f, df)


def testPower(v):
    def f(v):
        return v ** 3

    def df(v):
        return 3 * v ** 2
    _test(f, df)


def testDivision(v):
    def f(v):
        return v / 3.0

    def df(v):
        return as_ufl(1.0 / 3.0)
    _test(f, df)


def testDivision2(v):
    def f(v):
        return 3.0 / v

    def df(v):
        return -3.0 / v ** 2
    _test(f, df)


def testExp(v):
    def f(v):
        return exp(v)

    def df(v):
        return exp(v)
    _test(f, df)


def testLn(v):
    def f(v):
        return ln(v)

    def df(v):
        return 1.0 / v
    _test(f, df)


def testSin(v):
    def f(v):
        return sin(v)

    def df(v):
        return cos(v)
    _test(f, df)


def testCos(v):
    def f(v):
        return cos(v)

    def df(v):
        return -sin(v)
    _test(f, df)


def testTan(v):
    def f(v):
        return tan(v)

    def df(v):
        return 2.0 / (cos(2.0 * v) + 1.0)
    _test(f, df)

# TODO: Check the following tests. They run into strange math domain errors.
# def testAsin(v):
#    def f(v):  return asin(v)
#    def df(v): return 1/sqrt(1.0 - v**2)
#    _test(f, df)

# def testAcos(v):
#    def f(v):  return acos(v)
#    def df(v): return -1/sqrt(1.0 - v**2)
#    _test(f, df)


def testAtan(v):
    def f(v):
        return atan(v)

    def df(v):
        return 1 / (1.0 + v ** 2)
    _test(f, df)


def testIndexSum(v):
    def f(v):
        # 3*v + 4*v**2 + 5*v**3
        a = as_vector((v, v ** 2, v ** 3))
        b = as_vector((3, 4, 5))
        i, = indices(1)
        return a[i] * b[i]

    def df(v):
        return 3 + 4 * 2 * v + 5 * 3 * v ** 2
    _test(f, df)


def testCoefficient():
    coord_elem = VectorElement("P", triangle, 1, dim=3)
    mesh = Mesh(coord_elem)
    V = FunctionSpace(mesh, FiniteElement("P", triangle, 1))
    v = Coefficient(V)
    assert round(expand_derivatives(diff(v, v))-1.0, 7) == 0


def testDiffX():
    cell = triangle
    x = SpatialCoordinate(cell)
    f = x[0] ** 2 * x[1] ** 2
    i, = indices(1)
    df1 = diff(f, x)
    df2 = as_vector(f.dx(i), i)

    xv = (2, 3)
    df10 = df1[0](xv)
    df11 = df1[1](xv)
    df20 = df2[0](xv)
    df21 = df2[1](xv)
    assert round(df10 - df20, 7) == 0
    assert round(df11 - df21, 7) == 0
    assert round(df10 - 2 * 2 * 9, 7) == 0
    assert round(df11 - 2 * 4 * 3, 7) == 0

# TODO: More tests involving wrapper types and indices
