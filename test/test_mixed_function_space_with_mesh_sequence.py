from ufl import (
    CellVolume,
    Coefficient,
    FacetArea,
    FacetNormal,
    FunctionSpace,
    Measure,
    Mesh,
    MeshSequence,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    grad,
    inner,
    split,
    triangle,
)
from ufl.algorithms import compute_form_data
from ufl.domain import extract_domains
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pullback import contravariant_piola, identity_pullback
from ufl.sobolevspace import H1, L2, HDiv


def test_mixed_function_space_with_mesh_sequence_basic():
    cell = triangle
    elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2])
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=101)
    mesh2 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=102)
    domain = MeshSequence([mesh0, mesh1, mesh2])
    V = FunctionSpace(domain, elem)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    u0, u1, u2 = split(u)
    v0, v1, v2 = split(v)
    f0, f1, f2 = split(f)
    g0, g1, g2 = split(g)
    dx1 = Measure("dx", mesh1)
    x = SpatialCoordinate(mesh1)
    form = x[1] * f0 * inner(grad(u0), v1) * dx1(999)
    fd = compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=True,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(CellVolume, FacetArea),
        do_apply_restrictions=True,
        do_estimate_degrees=True,
        complex_mode=False,
    )
    (id0,) = fd.integral_data
    assert fd.preprocessed_form.arguments() == (v, u)
    assert fd.reduced_coefficients == [f]
    assert form.coefficients()[fd.original_coefficient_positions[0]] is f
    assert id0.domain is mesh1
    assert id0.integral_type == "cell"
    assert id0.subdomain_id == (999,)
    assert fd.original_form.domain_numbering()[id0.domain] == 0
    assert id0.integral_coefficients == set([f])
    assert id0.enabled_coefficients == [True]


def test_mixed_function_space_with_mesh_sequence_signature():
    cell = triangle
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=101)
    dx0 = Measure("dx", mesh0)
    dx1 = Measure("dx", mesh1)
    n0 = FacetNormal(mesh0)
    n1 = FacetNormal(mesh1)
    form_a = inner(n1, n1) * dx0(999)
    form_b = inner(n0, n0) * dx1(999)
    assert form_a.signature() == form_b.signature()
    assert extract_domains(form_a) == (mesh0, mesh1)
    assert extract_domains(form_b) == (mesh1, mesh0)


def test_mixed_function_space_with_mesh_sequence_hash():
    cell = triangle
    elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2])
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=101)
    mesh2 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=102)
    domain = MeshSequence([mesh0, mesh1, mesh2])
    domain_ = MeshSequence([mesh0, mesh1, mesh2])
    V = FunctionSpace(domain, elem)
    V_ = FunctionSpace(domain_, elem)
    u = TrialFunction(V)
    u_ = TrialFunction(V_)
    assert hash(domain_) == hash(domain)
    assert domain_ == domain
    assert hash(V_) == hash(V)
    assert V_ == V
    assert hash(u_) == hash(u)
    assert u_ == u
