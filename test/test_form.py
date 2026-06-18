import pytest
from utils import LagrangeElement

from ufl import (
    Argument,
    Coefficient,
    Cofunction,
    Form,
    FormProduct,
    FormSum,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    action,
    derivative,
    dot,
    ds,
    dx,
    grad,
    inner,
    nabla_grad,
    replace,
    triangle,
)
from ufl.form import BaseForm


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

    f = action(-v, u)
    df = derivative(9 * f, u)
    assert isinstance(f, FormSum)
    assert f.weights()[0] == -1
    assert isinstance(df, FormSum)
    assert df.weights()[0] == -9


def test_form_product_constructor_and_arguments(domain):
    element = LagrangeElement(triangle, 1)
    V = FunctionSpace(domain, element)
    v = TestFunction(V)
    f = Coefficient(V)
    g = Coefficient(V)
    h = Coefficient(V)

    Lf = f * v * dx
    Lg = g * v * dx
    Lh = h * v * dx

    product = FormProduct(Lf, Lg)
    assert isinstance(product, BaseForm)
    assert product.factors() == (Lf, Lg)
    assert product.ufl_operands == (Lf, Lg)
    assert product.factor_arguments() == (Lf.arguments(), Lg.arguments())

    arguments = product.arguments()
    assert tuple(argument.number() for argument in arguments) == (0, 1)
    assert tuple(argument.part() for argument in arguments) == (None, None)
    assert tuple(argument.ufl_function_space() for argument in arguments) == (V, V)
    assert Lg.arguments()[0].number() == 0

    assert product.coefficients() == (f, g)
    assert product.ufl_domains() == (domain,)

    nested = FormProduct(Lf, FormProduct(Lg, Lh))
    assert nested.factors() == (Lf, Lg, Lh)
    assert tuple(argument.number() for argument in nested.arguments()) == (0, 1, 2)


def test_form_product_rejects_invalid_inputs(domain):
    element = LagrangeElement(triangle, 1)
    V = FunctionSpace(domain, element)
    v = TestFunction(V)
    f = Coefficient(V)
    L = f * v * dx

    with pytest.raises(ValueError):
        FormProduct(L)
    with pytest.raises(TypeError):
        FormProduct(L, 1)


def test_form_product_is_explicit_not_mul_overload(domain):
    element = LagrangeElement(triangle, 1)
    V = FunctionSpace(domain, element)
    v = TestFunction(V)
    f = Coefficient(V)
    g = Coefficient(V)
    Lf = f * v * dx
    Lg = g * v * dx

    with pytest.raises(TypeError):
        Lf * Lg


def test_form_product_replace(domain):
    element = LagrangeElement(triangle, 1)
    V = FunctionSpace(domain, element)
    v = TestFunction(V)
    f = Coefficient(V)
    g = Coefficient(V)

    Lf = f * v * dx
    Lg = g * v * dx
    product = FormProduct(Lf, Lg)
    replaced = replace(product, {f: g})

    assert isinstance(replaced, FormProduct)
    assert bool(replaced.factors()[0] == Lg)
    assert bool(replaced.factors()[1] == Lg)
    assert tuple(argument.number() for argument in replaced.arguments()) == (0, 1)


def test_form_product_derivative_product_rule(domain):
    element = LagrangeElement(triangle, 1)
    V = FunctionSpace(domain, element)
    v = TestFunction(V)
    f = Coefficient(V)
    direction = Argument(V, 1)

    L = f * v * dx
    product = FormProduct(L, L)
    dproduct = derivative(product, f, direction)

    assert isinstance(dproduct, FormSum)
    assert len(dproduct.components()) == 2
    assert all(isinstance(component, FormProduct) for component in dproduct.components())

    dL = derivative(L, f, direction)
    expected_first = FormProduct(dL, L)
    expected_second = FormProduct(L, dL)
    assert bool(dproduct.components()[0] == expected_first)
    assert bool(dproduct.components()[1] == expected_second)


def test_form_product_exported_from_classes():
    from ufl.classes import FormProduct as ClassesFormProduct

    assert ClassesFormProduct is FormProduct
