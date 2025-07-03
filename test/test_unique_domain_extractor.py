from utils import FiniteElement, LagrangeElement, MixedElement

import pytest
from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    MeshSequence,
    SpatialCoordinate,
    Measure,
    TrialFunction,
    split,
    triangle,
    cos,
    inner
)

from ufl.domain import extract_unique_domain_dag
from ufl.pullback import contravariant_piola, identity_pullback
from ufl.sobolevspace import L2, HDiv

def test_extract_unique_domain():
    cell = triangle
    elem0 = LagrangeElement(cell, 1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 2, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 1, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2])
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=100)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=101)
    mesh3 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=102)
    domain = MeshSequence([mesh1, mesh2, mesh3])
    V = FunctionSpace(domain, elem)

    u = TrialFunction(V)
    u1, u2, u3 = split(u)
    for i, u_i in enumerate((u1, u2, u3)):
        assert extract_unique_domain_dag(u_i) == domain[i]

    f = Coefficient(V)
    f1, f2, f3 = split(f)
    for i, f_i in enumerate((f1, f2, f3)):
        assert extract_unique_domain_dag(f_i) == domain[i]

    x1, y1 = SpatialCoordinate(mesh1)
    expr = u1 + x1 * cos(x1)
    assert extract_unique_domain_dag(expr) == mesh1

    x2, y2 = SpatialCoordinate(mesh2)
    with pytest.raises(ValueError) as e_info:
        _ = extract_unique_domain_dag(u1 + u2)
        _ = extract_unique_domain_dag(u1 + u2 + x2 * cos(x2 * u1))


def test_extract_unique_domain_form():
    cell = triangle
    elem0 = LagrangeElement(cell, 1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 2, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 1, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2])
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=100)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=101)
    mesh3 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=102)
    domain = MeshSequence([mesh1, mesh2, mesh3])
    V = FunctionSpace(domain, elem)

    u = TrialFunction(V)
    u1, u2, u3 = split(u)

    f = Coefficient(V)
    f1, f2, f3 = split(f)

    dx1 = Measure("dx", mesh1)

    form1 = inner(u1, f1) * dx1

    assert extract_unique_domain_dag(form1) == mesh1