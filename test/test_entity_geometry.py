"""Tests that geometry of a sub-entity only appears in integrals over it."""

import pytest
from utils import LagrangeElement

from ufl import (
    Coefficient,
    FacetNormal,
    FunctionSpace,
    Measure,
    Mesh,
    TestFunction,
    interval,
    tetrahedron,
)
from ufl.algorithms import compute_form_data
from ufl.geometry import CellRidgeJacobian, ReferenceRidgeVolume


@pytest.fixture
def domain():
    return Mesh(LagrangeElement(tetrahedron, 1, (3,)))


def _form(domain, quantity, integral_type):
    v = TestFunction(FunctionSpace(domain, LagrangeElement(tetrahedron, 1)))
    integrand = quantity * v
    if integral_type == "interior_facet":
        # Otherwise the missing restriction is reported before the geometry
        integrand = integrand("+")
    return integrand * Measure(integral_type, domain=domain)


@pytest.mark.parametrize("quantity", [lambda d: CellRidgeJacobian(d)[0, 0], ReferenceRidgeVolume])
def test_ridge_geometry_allowed_in_ridge_integral(domain, quantity):
    compute_form_data(_form(domain, quantity(domain), "ridge"))


_RIDGE_QUANTITIES = [
    (lambda d: CellRidgeJacobian(d)[0, 0], "CellRidgeJacobian"),
    (ReferenceRidgeVolume, "ReferenceRidgeVolume"),
]


@pytest.mark.parametrize("integral_type", ["cell", "exterior_facet", "vertex"])
@pytest.mark.parametrize("quantity,name", _RIDGE_QUANTITIES)
def test_ridge_geometry_rejected_outside_ridge_integral(domain, integral_type, quantity, name):
    form = _form(domain, quantity(domain), integral_type)
    message = f"Integral of type {integral_type} cannot contain a {name}"
    with pytest.raises(ValueError, match=message):
        compute_form_data(form)


@pytest.mark.parametrize("quantity,name", _RIDGE_QUANTITIES)
def test_ridge_geometry_rejected_in_interior_facet_integral(domain, quantity, name):
    # Rejected before the geometry check, by `apply_restrictions`, which has
    # no restriction rule for ridge geometry
    form = _form(domain, quantity(domain), "interior_facet")
    with pytest.raises(ValueError, match=name):
        compute_form_data(form)


@pytest.mark.parametrize("integral_type", ["exterior_facet", "interior_facet"])
def test_facet_geometry_allowed_in_facet_integral(domain, integral_type):
    compute_form_data(_form(domain, FacetNormal(domain)[0], integral_type))


@pytest.mark.parametrize("integral_type", ["cell", "ridge", "vertex"])
def test_facet_geometry_rejected_outside_facet_integral(domain, integral_type):
    form = _form(domain, FacetNormal(domain)[0], integral_type)
    with pytest.raises(ValueError, match=f"Integral of type {integral_type} cannot contain a"):
        compute_form_data(form)


def test_ridge_geometry_with_coefficient_on_ridge_mesh(domain):
    """A ridge integral may mix parent ridge geometry with ridge-mesh data.

    The check looks up each quantity's integral type through its own
    domain, so the coefficient on the codimension-2 mesh must not be
    mistaken for ridge geometry, nor the parent's ridge geometry rejected.
    """
    ridge_mesh = Mesh(LagrangeElement(interval, 1, (3,)))
    u = Coefficient(FunctionSpace(ridge_mesh, LagrangeElement(interval, 1)))
    form = u * CellRidgeJacobian(domain)[0, 0] * Measure("ridge", domain=domain)
    compute_form_data(form)
