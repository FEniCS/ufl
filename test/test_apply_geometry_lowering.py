"""Tests for geometry lowering of a coefficient defined on a companion domain.

A companion domain is a domain distinct from the integration domain that
names one of its facets/ridges (e.g. the domain of a coefficient defined
on a submesh, integrated via a measure on the parent mesh).
It's Jacobian/SpatialCoordinate is rewritten in the terms of the integration domain's
own Jacobian/SpatialCoordinate, rather than lowered directly.
This avoids having to pack the same geometric information twice.
"""

import pytest
from utils import LagrangeElement

from ufl import (
    Coefficient,
    Dx,
    FunctionSpace,
    Measure,
    Mesh,
    SpatialCoordinate,
    hexahedron,
    interval,
    quadrilateral,
    tetrahedron,
    triangle,
)
from ufl.algorithms.analysis import extract_type
from ufl.algorithms.compute_form_data import compute_form_data
from ufl.classes import (
    CellFacetJacobian,
    CellRidgeJacobian,
    FacetJacobian,
    Jacobian,
    JacobianInverse,
    RidgeJacobian,
)
from ufl.domain import extract_unique_domain


def _lower(form, preserve_geometry_types=(Jacobian,)):
    """Lower a form the same way FFCx does (see ffcx/analysis.py)."""
    fd = compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=True,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=preserve_geometry_types,
        do_apply_restrictions=True,
        do_estimate_degrees=False,
        complex_mode=False,
    )
    (integral_data,) = fd.integral_data
    (integral,) = integral_data.integrals
    return integral.integrand()


def test_grad_of_facet_companion_uses_facet_jacobian():
    """Grad(u) of a codim-1 companion (facet) domain coefficient must not
    reference that domain's own Jacobian/JacobianInverse.
    """
    domain = Mesh(LagrangeElement(quadrilateral, 1, (2,)))
    codomain = Mesh(LagrangeElement(interval, 1, (2,)))
    space = FunctionSpace(codomain, LagrangeElement(interval, 1))
    u = Coefficient(space)
    ds = Measure("ds", domain=domain)

    integrand = _lower(Dx(u, 0) * ds)

    for cls in (Jacobian, JacobianInverse):
        for obj in extract_type(integrand, cls):
            assert extract_unique_domain(obj) != codomain

    facet_jacobians = extract_type(integrand, (FacetJacobian, CellFacetJacobian))
    assert facet_jacobians
    for obj in facet_jacobians:
        assert extract_unique_domain(obj) == domain


def test_bare_spatial_coordinate_of_facet_companion_uses_parent_field():
    """A bare SpatialCoordinate of a codim-1 companion domain names the same
    physical points as the integration domain's own facet, so it must be
    rewritten in terms of the integration domain's SpatialCoordinate.
    """
    domain = Mesh(LagrangeElement(hexahedron, 1, (3,)))
    codomain = Mesh(LagrangeElement(quadrilateral, 1, (3,)))
    x = SpatialCoordinate(codomain)
    ds = Measure("ds", domain=domain)

    integrand = _lower(x[0] * ds)

    for obj in extract_type(integrand, SpatialCoordinate):
        assert extract_unique_domain(obj) == domain


def test_grad_of_ridge_companion_uses_ridge_jacobian():
    """Grad(u) of a codim-2 companion (ridge) domain coefficient must not
    reference that domain's own Jacobian/JacobianInverse.
    """
    domain = Mesh(LagrangeElement(tetrahedron, 1, (3,)))
    codomain = Mesh(LagrangeElement(interval, 1, (3,)))
    space = FunctionSpace(codomain, LagrangeElement(interval, 1))
    u = Coefficient(space)
    dr = Measure("ridge", domain=domain)

    integrand = _lower(Dx(u, 0) * dr)

    for cls in (Jacobian, JacobianInverse):
        for obj in extract_type(integrand, cls):
            assert extract_unique_domain(obj) != codomain

    ridge_jacobians = extract_type(integrand, (RidgeJacobian, CellRidgeJacobian))
    assert ridge_jacobians
    for obj in ridge_jacobians:
        assert extract_unique_domain(obj) == domain


def test_codim0_companion_domain_untouched():
    """A same-topological-dimension companion domain (e.g. DOLFINx's
    already-working codim-0 submesh case) must be left exactly as before
    this fix: its own Jacobian is still used directly.
    """
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    codomain = Mesh(LagrangeElement(triangle, 1, (2,)))
    space = FunctionSpace(codomain, LagrangeElement(triangle, 1))
    u = Coefficient(space)
    dx = Measure("dx", domain=domain)

    integrand = _lower(Dx(u, 0) * dx)

    jacobians = extract_type(integrand, (Jacobian, JacobianInverse))
    assert any(extract_unique_domain(obj) == codomain for obj in jacobians)


def test_facet_companion_in_cell_integral_raises():
    """A codim-1 companion domain only makes sense inside a facet integral
    -- there is no local facet index to relate it to the integration
    domain's own geometry inside a cell integral.
    """
    domain = Mesh(LagrangeElement(quadrilateral, 1, (2,)))
    codomain = Mesh(LagrangeElement(interval, 1, (2,)))
    space = FunctionSpace(codomain, LagrangeElement(interval, 1))
    u = Coefficient(space)
    dx = Measure("dx", domain=domain)

    with pytest.raises(NotImplementedError):
        _lower(Dx(u, 0) * dx)


def test_companion_domain_gdim_mismatch_raises():
    """A companion domain embedded in a different geometric dimension than
    the integration domain cannot name one of its facets/ridges.
    """
    domain = Mesh(LagrangeElement(quadrilateral, 1, (2,)))
    codomain = Mesh(LagrangeElement(interval, 1, (3,)))
    space = FunctionSpace(codomain, LagrangeElement(interval, 1))
    u = Coefficient(space)
    ds = Measure("ds", domain=domain)

    with pytest.raises(ValueError):
        _lower(Dx(u, 0) * ds)
