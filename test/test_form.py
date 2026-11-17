import pytest
from utils import LagrangeElement, RealElement

from ufl import (
    Action,
    Coargument,
    Coefficient,
    Cofunction,
    Constant,
    Form,
    FormSum,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    action,
    adjoint,
    derivative,
    dot,
    ds,
    dx,
    grad,
    inner,
    nabla_grad,
    triangle,
)
from ufl.algorithms.restructure_base_form import restructure_base_form
from ufl.form import BaseForm, ZeroBaseForm


@pytest.fixture
def element():
    cell = triangle
    element = LagrangeElement(cell, 1)
    return element


@pytest.fixture
def domain():
    cell = triangle
    return Mesh(LagrangeElement(cell, 1, (2,)))


@pytest.fixture
def mass(domain):
    cell = triangle
    element = LagrangeElement(cell, 1)
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    return u * v * dx


@pytest.fixture
def stiffness(domain):
    cell = triangle
    element = LagrangeElement(cell, 1)
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    return inner(grad(u), grad(v)) * dx


@pytest.fixture
def convection(domain):
    cell = triangle
    element = LagrangeElement(cell, 1, (2,))
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    u = TrialFunction(space)
    w = Coefficient(space)
    return dot(dot(w, nabla_grad(u)), v) * dx


@pytest.fixture
def load(domain):
    cell = triangle
    element = LagrangeElement(cell, 1)
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    v = TestFunction(space)
    return f * v * dx


@pytest.fixture
def boundary_load(domain):
    cell = triangle
    element = LagrangeElement(cell, 1)
    space = FunctionSpace(domain, element)
    f = Coefficient(space)
    v = TestFunction(space)
    return f * v * ds


def test_form_arguments(mass, stiffness, convection, load):
    v, u = mass.arguments()
    (f,) = load.coefficients()

    assert v.number() == 0
    assert u.number() == 1
    assert stiffness.arguments() == (v, u)
    assert load.arguments() == (v,)

    assert (v * dx).arguments() == (v,)
    assert (v * dx + v * ds).arguments() == (v,)
    assert (v * dx + f * v * ds).arguments() == (v,)
    assert (u * v * dx(1) + v * u * dx(2)).arguments() == (v, u)
    assert ((f * v) * u * dx + (u * 3) * (v / 2) * dx(2)).arguments() == (v, u)


def test_form_coefficients(element, domain):
    space = FunctionSpace(domain, element)
    v = TestFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)

    assert (g * dx).coefficients() == (g,)
    assert (g * dx + g * ds).coefficients() == (g,)
    assert (g * dx + f * ds).coefficients() == (f, g)
    assert (g * dx(1) + f * dx(2)).coefficients() == (f, g)
    assert (g * v * dx + f * v * dx(2)).coefficients() == (f, g)


def test_form_domains():
    cell = triangle
    element = LagrangeElement(cell, 1)
    domain = Mesh(LagrangeElement(cell, 1, (2,)))
    V = FunctionSpace(domain, element)

    v = TestFunction(V)
    f = Coefficient(V)
    x = SpatialCoordinate(domain)[0]

    assert (x * dx).ufl_domains() == (domain,)
    assert (v * dx).ufl_domains() == (domain,)
    assert (f * dx).ufl_domains() == (domain,)
    assert (x * v * f * dx).ufl_domains() == (domain,)
    assert (1 * dx(domain)).ufl_domains() == (domain,)


def test_form_empty(mass):
    assert not mass.empty()
    assert Form([]).empty()


def test_form_integrals(mass, boundary_load):
    assert isinstance(mass.integrals(), tuple)
    assert len(mass.integrals()) == 1
    assert mass.integrals()[0].integral_type() == "cell"
    assert mass.integrals_by_type("cell") == mass.integrals()
    assert mass.integrals_by_type("exterior_facet") == ()
    assert isinstance(boundary_load.integrals_by_type("cell"), tuple)
    assert len(boundary_load.integrals_by_type("cell")) == 0
    assert len(boundary_load.integrals_by_type("exterior_facet")) == 1


def test_form_call():
    element = LagrangeElement(triangle, 1)
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, element)
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Coefficient(V)
    g = Coefficient(V)
    a = g * inner(grad(v), grad(u)) * dx
    M = a(f, f, coefficients={g: 1})
    assert M == grad(f) ** 2 * dx

    import sys

    if sys.version_info.major >= 3 and sys.version_info.minor >= 5:
        a = u * v * dx
        M = eval("(a @ f) @ g")
        assert M == g * f * dx


def test_formsum(mass):
    element = LagrangeElement(triangle, 1)
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    V = FunctionSpace(domain, element)
    v = Cofunction(V.dual())
    u = Coefficient(V)

    assert v + mass
    assert mass + v
    assert isinstance((mass + v), FormSum)

    assert len((mass + v + v).components()) == 3
    # Variational forms are summed appropriately
    assert len((mass + v + mass).components()) == 2

    assert v - mass
    assert mass - v
    assert isinstance((mass + v), FormSum)

    assert -v
    assert isinstance(-v, BaseForm)
    assert (-v).weights()[0] == -1

    assert 2 * v
    assert isinstance(2 * v, BaseForm)
    assert (2 * v).weights()[0] == 2

    f = restructure_base_form(action(-v, u))
    df = derivative(9 * f, u)
    assert f == Action(-v, u)
    assert isinstance(df, FormSum)
    assert df.weights()[0] == 9

    assert FormSum((mass, 1)) is mass


def test_form_divided_by_constant_scales(mass):
    V = mass.arguments()[0].ufl_function_space()
    c = Constant(V.ufl_domain())
    cofunction = Cofunction(V.dual())
    for form in (mass, mass + cofunction, cofunction, Action(mass, Coefficient(V))):
        assert form / 2 == 0.5 * form
        assert form / c == (1 / c) * form
        with pytest.raises(TypeError):
            form / Coefficient(V)
        with pytest.raises(TypeError):
            2 / form


def test_constant_scales_a_form_from_either_side(mass):
    V = mass.arguments()[0].ufl_function_space()
    c = Constant(V.ufl_domain())
    u = Coefficient(V)
    forms = (
        mass,
        u * TestFunction(V) * dx,
        Cofunction(V.dual()),
        Coargument(V.dual(), 0),
        Action(mass, u),
        adjoint(mass),
        ZeroBaseForm(mass.arguments()),
    )
    for form in forms:
        for scalar in (2, 0.5, c, 2 * c + 1):
            assert form * scalar == scalar * form


def test_form_acts_on_real_function_on_the_right(domain):
    """A function in a Real space is constant, but on the right a form acts on it."""
    V = FunctionSpace(domain, LagrangeElement(triangle, 1))
    R = FunctionSpace(domain, RealElement(triangle))
    r, c = Coefficient(R), Constant(domain)
    v = TestFunction(V)
    A = inner(TrialFunction(R), v) * dx

    assert (A @ r).arguments() == (v,)
    with pytest.warns(DeprecationWarning, match="form @ f"):
        assert A * r == A @ r
    # A constant scales, also on the right
    assert (A * c).arguments() == A.arguments()
    # An expression of a Real function could be acted on or scale, so it raises
    # on the right, and scales on the left.
    for operand in (2 * r, r * r, c * r):
        with pytest.raises(TypeError, match="multiply it from the left"):
            A * operand
        assert (operand * A).arguments() == A.arguments()


def test_matmul_only_acts(mass):
    V = mass.arguments()[0].ufl_function_space()
    u, c = Coefficient(V), Constant(V.ufl_domain())
    assert mass @ u == action(mass, u)
    for operand in (2 * u, c):
        with pytest.raises(TypeError, match="can only act on"):
            mass @ operand
    with pytest.raises(TypeError):
        mass @ 2
