__authors__ = "Martin Sandve Alnæs"
__date__ = "2008-03-12 -- 2009-01-28"

import pytest
from utils import FiniteElement, LagrangeElement, MixedElement

from ufl import (
    Argument,
    Coefficient,
    Coefficients,
    FacetNormal,
    Form,
    FunctionSpace,
    Mesh,
    SpatialCoordinate,
    TensorProductCell,
    atan2,
    bessel_J,
    bessel_K,
    cell_avg,
    cofac,
    conditional,
    cos,
    derivative,
    det,
    dev,
    diff,
    div,
    dot,
    dx,
    facet_avg,
    grad,
    i,
    inner,
    interval,
    inv,
    lt,
    max_value,
    min_value,
    nabla_div,
    nabla_grad,
    quadrilateral,
    sin,
    skew,
    sym,
    tan,
    tr,
    triangle,
    variable,
)
from ufl.algorithms import estimate_total_polynomial_degree
from ufl.classes import CellCoordinate
from ufl.core.external_operator import ExternalOperator
from ufl.core.interpolate import Interpolate
from ufl.pullback import identity_pullback
from ufl.sobolevspace import H1, L2


def test_total_degree_estimation():
    V1 = LagrangeElement(triangle, 1)
    V2 = LagrangeElement(triangle, 2)
    VV = LagrangeElement(triangle, 3, (2,))
    VM = MixedElement([V1, V2])

    domain = Mesh(LagrangeElement(triangle, 1, (2,)))

    v1_space = FunctionSpace(domain, V1)
    v2_space = FunctionSpace(domain, V2)
    vv_space = FunctionSpace(domain, VV)
    vm_space = FunctionSpace(domain, VM)

    v1 = Argument(v1_space, 2)
    v2 = Argument(v2_space, 3)
    f1, f2 = Coefficients(vm_space)
    vv = Argument(vv_space, 4)
    vu = Argument(vv_space, 5)

    x, y = SpatialCoordinate(domain)
    assert estimate_total_polynomial_degree(x) == 1
    assert estimate_total_polynomial_degree(x * y) == 2
    assert estimate_total_polynomial_degree(x**3) == 3
    assert estimate_total_polynomial_degree(x**3) == 3
    assert estimate_total_polynomial_degree((x - 1) ** 4) == 4

    assert estimate_total_polynomial_degree(vv[0]) == 3
    assert estimate_total_polynomial_degree(v2 * vv[0]) == 5
    assert estimate_total_polynomial_degree(vu[0] * vv[0]) == 6
    assert estimate_total_polynomial_degree(vu[i] * vv[i]) == 6

    assert estimate_total_polynomial_degree(v1) == 1
    assert estimate_total_polynomial_degree(Interpolate(Coefficient(v1_space), v2_space)) == 2
    assert estimate_total_polynomial_degree(v2) == 2

    # f1 lives on the mixed element's degree-1 sub-element, so its
    # degree is 1, not the mixed element's max of 2.
    assert estimate_total_polynomial_degree(f1) == 1

    assert estimate_total_polynomial_degree(f2) == 2
    assert estimate_total_polynomial_degree(v2 * v1) == 3

    # f1's own degree is 1 (see above), so this is 1 + 1 = 2, not 3.
    assert estimate_total_polynomial_degree(f1 * v1) == 2

    assert estimate_total_polynomial_degree(f2 * v1) == 3
    assert estimate_total_polynomial_degree(f2 * v2 * v1) == 5

    assert estimate_total_polynomial_degree(f2 + 3) == 2
    assert estimate_total_polynomial_degree(f2 * 3) == 2
    assert estimate_total_polynomial_degree(f2**3) == 6
    assert estimate_total_polynomial_degree(f2 / 3) == 2
    assert estimate_total_polynomial_degree(f2 / v2) == 4
    assert estimate_total_polynomial_degree(f2 / (x - 1)) == 3

    assert estimate_total_polynomial_degree(v1.dx(0)) == 0
    assert estimate_total_polynomial_degree(f2.dx(0)) == 1

    assert estimate_total_polynomial_degree(f2 * v2.dx(0) * v1.dx(0)) == 2 + 1

    assert estimate_total_polynomial_degree(f2) == 2
    assert estimate_total_polynomial_degree(f2**2) == 4
    assert estimate_total_polynomial_degree(f2**3) == 6
    assert estimate_total_polynomial_degree(f2**3 * v1) == 7
    assert estimate_total_polynomial_degree(f2**3 * v1 + f1 * v1) == 7

    # Math functions of constant values are constant values
    nx, _ny = FacetNormal(domain)
    e = nx**2
    for f in [sin, cos, tan, abs, lambda z: z**7]:
        assert estimate_total_polynomial_degree(f(e)) == 0

    # Based on the arbitrary chosen math function heuristics...
    heuristic_add = 2
    e = x**3
    for f in [sin, cos, tan]:
        assert estimate_total_polynomial_degree(f(e)) == 3 + heuristic_add


def test_tensor_product_degree_estimation():
    cell = TensorProductCell(quadrilateral, interval)
    domain = Mesh(LagrangeElement(cell, 1, (3,)))
    space = FunctionSpace(domain, LagrangeElement(cell, 7))
    u = Argument(space, 1)
    v = Argument(space, 2)

    assert estimate_total_polynomial_degree(u) == 7
    assert estimate_total_polynomial_degree(u * v) == 14

    assert estimate_total_polynomial_degree(u.dx(0)) == 7
    assert estimate_total_polynomial_degree(inner(grad(u), grad(v))) == 14

    x = SpatialCoordinate(domain)
    assert estimate_total_polynomial_degree(x[0]) == 1


def test_some_compound_types():
    # NB! Although some compound types are supported here,
    # some derivatives and compounds must be preprocessed
    # prior to degree estimation. In generic code, this algorithm
    # should only be applied after preprocessing.

    etpd = estimate_total_polynomial_degree

    P2 = LagrangeElement(triangle, 2)
    V2 = LagrangeElement(triangle, 2, (2,))
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))

    u = Coefficient(FunctionSpace(domain, P2))
    v = Coefficient(FunctionSpace(domain, V2))

    assert etpd(u.dx(0)) == 2 - 1
    assert etpd(grad(u)) == 2 - 1
    assert etpd(nabla_grad(u)) == 2 - 1
    assert etpd(div(u)) == 2 - 1

    assert etpd(v.dx(0)) == 2 - 1
    assert etpd(grad(v)) == 2 - 1
    assert etpd(nabla_grad(v)) == 2 - 1
    assert etpd(div(v)) == 2 - 1
    assert etpd(nabla_div(v)) == 2 - 1

    assert etpd(dot(v, v)) == 2 + 2
    assert etpd(inner(v, v)) == 2 + 2

    assert etpd(dot(grad(u), grad(u))) == 2 - 1 + 2 - 1
    assert etpd(inner(grad(u), grad(u))) == 2 - 1 + 2 - 1

    assert etpd(dot(grad(v), grad(v))) == 2 - 1 + 2 - 1
    assert etpd(inner(grad(v), grad(v))) == 2 - 1 + 2 - 1


@pytest.mark.parametrize("degree", [1, 3])
def test_terminal_degree_estimation(degree):
    """Rules for terminals that no other test exercises."""
    domain = Mesh(LagrangeElement(triangle, degree, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree)))

    # A cell coordinate provides one additional degree
    assert estimate_total_polynomial_degree(CellCoordinate(domain)[0]) == 1
    assert estimate_total_polynomial_degree(CellCoordinate(domain)[0] * u) == 1 + degree


@pytest.mark.parametrize("degree1", [1, 3])
@pytest.mark.parametrize("degree2", [1, 3])
def test_averaging_degree_estimation(degree1, degree2):
    """Cell and facet averages are cellwise constant."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    v2_space = FunctionSpace(domain, LagrangeElement(triangle, degree1))
    v3_space = FunctionSpace(domain, LagrangeElement(triangle, degree2))
    u = Coefficient(v2_space)
    w = Coefficient(v3_space)

    assert estimate_total_polynomial_degree(cell_avg(u)) == 0
    assert estimate_total_polynomial_degree(facet_avg(u)) == 0
    assert estimate_total_polynomial_degree(cell_avg(u) * w) == degree2
    assert estimate_total_polynomial_degree(facet_avg(u) * w) == degree2


@pytest.mark.parametrize("degree1", [1, 3])
@pytest.mark.parametrize("degree2", [1, 3])
def test_abs_degree_estimation(degree1, degree2):
    """Abs leaves the degree of its operand unchanged."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree1)))
    w = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree2)))

    assert estimate_total_polynomial_degree(abs(u)) == degree1
    assert estimate_total_polynomial_degree(abs(u) * w) == degree1 + degree2


@pytest.mark.parametrize("degree1", [1, 3])
@pytest.mark.parametrize("degree2", [1, 3])
def test_conditional_degree_estimation(degree1, degree2):
    """The condition carries no degree; the branches are combined with max."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree1)))
    w = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree2)))
    x = SpatialCoordinate(domain)

    # The degree of the condition does not contribute
    assert estimate_total_polynomial_degree(conditional(lt(u, w), u, w)) == max(degree1, degree2)
    assert estimate_total_polynomial_degree(conditional(lt(x[0], 0), u * w, u)) == max(
        degree1 + degree2, degree1
    )

    # min/max follow the same rule as conditional
    assert estimate_total_polynomial_degree(min_value(u, w)) == max(degree1, degree2)
    assert estimate_total_polynomial_degree(max_value(u, w)) == max(degree1, degree2)


@pytest.mark.parametrize("degree1", [1, 3])
@pytest.mark.parametrize("degree2", [1, 3])
def test_atan2_and_bessel_degree_estimation(degree1, degree2):
    """Both use the "+2 unless the argument is constant" heuristic."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree1)))
    w = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree2)))
    nx, ny = FacetNormal(domain)

    heuristic_add = 2
    assert estimate_total_polynomial_degree(atan2(u, w)) == max(degree1, degree2) + heuristic_add
    assert estimate_total_polynomial_degree(bessel_J(1, u)) == degree1 + heuristic_add

    # Of cellwise constant arguments they are constant
    assert estimate_total_polynomial_degree(atan2(nx, ny)) == 0
    assert estimate_total_polynomial_degree(bessel_K(1, nx * nx)) == 0


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("default_degree", [1, 4])
def test_default_degree_estimation(degree, default_degree):
    """Elements with no degree fall back to the given default degree."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    no_degree = FiniteElement("Custom", triangle, None, (), identity_pullback, L2)
    nodeg_space = FunctionSpace(domain, no_degree)
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree)))
    f = Coefficient(nodeg_space)

    etpd = estimate_total_polynomial_degree
    assert etpd(f, default_degree=default_degree) == default_degree
    assert etpd(f * u, default_degree=default_degree) == default_degree + degree

    # Interpolating into such an element uses the default degree too
    assert etpd(Interpolate(u, nodeg_space), default_degree=default_degree) == default_degree

    # The default degree is 1 when not given
    assert etpd(f) == 1


@pytest.mark.parametrize("degree1", [1, 3])
@pytest.mark.parametrize("degree2", [1, 3])
def test_coordinate_derivative_degree_estimation(degree1, degree2):
    """A shape derivative adds the degree of the deformation direction."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree1)))
    direction = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree2, (2,))))
    x = SpatialCoordinate(domain)

    # This also exercises the ExprList and ExprMapping rules, which hold the
    # operands of the CoordinateDerivative.
    form = derivative(u * dx, x, direction)
    assert estimate_total_polynomial_degree(form) == degree1 + degree2
    assert estimate_total_polynomial_degree(form.integrals()[0].integrand()) == degree1 + degree2


@pytest.mark.parametrize(
    "op, name",
    [
        (tr, "Trace"),
        (det, "Determinant"),
        (cofac, "Cofactor"),
        (inv, "Inverse"),
        (dev, "Deviatoric"),
        (skew, "Skew"),
        (sym, "Sym"),
    ],
)
@pytest.mark.parametrize("degree", [1, 3])
def test_unhandled_compounds_raise(op, name, degree):
    """Compounds that must be preprocessed away before degree estimation."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    t = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree, (2, 2))))

    with pytest.raises(ValueError, match=f"Missing degree handler for type {name}"):
        estimate_total_polynomial_degree(op(t))


@pytest.mark.parametrize("degree", [1, 3])
def test_unhandled_derivative_raises(degree):
    """Derivatives that must be applied before degree estimation."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree)))

    v = variable(u)
    with pytest.raises(ValueError, match="Missing degree handler for type VariableDerivative"):
        estimate_total_polynomial_degree(diff(v**2, v))


@pytest.mark.parametrize("degree", [1, 3])
def test_missing_handler_warns(degree):
    """Types with no rule at all warn and fall back to summing the operands."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    space = FunctionSpace(domain, LagrangeElement(triangle, degree))
    u = Coefficient(space)

    e = ExternalOperator(u, function_space=space)
    with pytest.warns(UserWarning, match="Missing degree estimation handler"):
        assert estimate_total_polynomial_degree(e) == degree


@pytest.mark.parametrize("degree", [1, 3])
@pytest.mark.parametrize("exponent", [1.5, 0.5, -2])
def test_non_integer_power_degree_estimation(degree, exponent):
    """A non-(positive integer) exponent falls back to the "+2" heuristic."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree)))

    heuristic_add = 2
    assert estimate_total_polynomial_degree(u**exponent) == degree + heuristic_add


@pytest.mark.parametrize("degree1", [1, 3])
@pytest.mark.parametrize("degree2", [1, 3])
def test_form_and_integral_input(degree1, degree2):
    """Forms, integrals and bare expressions are all accepted."""
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree1)))
    w = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, degree2)))

    form = u * w * dx
    assert estimate_total_polynomial_degree(form) == degree1 + degree2
    assert estimate_total_polynomial_degree(form.integrals()[0]) == degree1 + degree2
    assert estimate_total_polynomial_degree(form.integrals()[0].integrand()) == degree1 + degree2

    # The maximum is taken over the integrals of a form
    assert estimate_total_polynomial_degree(u * dx + form) == max(degree1, degree1 + degree2)


def test_form_without_integrals():
    with pytest.raises(ValueError, match=r"Form has no integrals\."):
        estimate_total_polynomial_degree(Form([]))


@pytest.mark.parametrize("degree", [(2, 1), (3, 2)])
def test_tuple_degree_estimation(degree):
    """Elements reporting a per-direction (tuple) degree."""
    cell = TensorProductCell(quadrilateral, interval)
    domain = Mesh(LagrangeElement(cell, 1, (3,)))
    element = FiniteElement("TensorProduct", cell, degree, (), identity_pullback, H1)
    space = FunctionSpace(domain, element)
    u = Coefficient(space)
    v = Coefficient(space)

    etpd = estimate_total_polynomial_degree
    assert etpd(u) == degree
    # Products add and sums take the max, componentwise
    assert etpd(u * v) == tuple(2 * d for d in degree)
    assert etpd(u + v) == degree
    assert etpd(u**3) == tuple(3 * d for d in degree)
    assert etpd(conditional(lt(u, v), u, v)) == degree
    # Scalars are broadcast over the tuple
    heuristic_add = 2
    assert etpd(sin(u)) == tuple(d + heuristic_add for d in degree)
    # The degree is not reduced for non-simplex cells
    assert etpd(grad(u)) == degree
