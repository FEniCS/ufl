from ufl import (triangle, Mesh, MixedMesh, FunctionSpace, TestFunction, TrialFunction, Coefficient, Constant,
                 Measure, SpatialCoordinate, FacetNormal, CellVolume, FacetArea, inner, grad, div, split, )
from ufl.algorithms import compute_form_data
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.pullback import identity_pullback, contravariant_piola
from ufl.sobolevspace import H1, HDiv, L2
from ufl.domain import extract_domains


def test_mixed_function_space_with_mixed_mesh_basic():
    cell = triangle
    elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1, (2, ), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2])
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=101)
    mesh2 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=102)
    domain = MixedMesh(mesh0, mesh1, mesh2)
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
    ds2 = Measure("ds", mesh2)
    x = SpatialCoordinate(mesh1)
    form = x[1] * f0 * inner(grad(u0), v1) * dx1(999) + div(f1) * g2 * inner(u1, grad(v2)) * ds2(888)
    fd = compute_form_data(form,
                           do_apply_function_pullbacks=True,
                           do_apply_integral_scaling=True,
                           do_apply_geometry_lowering=True,
                           preserve_geometry_types=(CellVolume, FacetArea),
                           do_apply_restrictions=True,
                           do_estimate_degrees=True,
                           complex_mode=False)
    id0, id1 = fd.integral_data
    assert fd.preprocessed_form.arguments() == (v, u)
    assert fd.reduced_coefficients == [f, g]
    assert form.coefficients()[fd.original_coefficient_positions[0]] is f
    assert form.coefficients()[fd.original_coefficient_positions[1]] is g
    assert id0.domain is mesh1
    assert id0.integral_type == 'cell'
    assert id0.subdomain_id == (999, )
    assert fd.original_form.domain_numbering()[id0.domain] == 0
    assert id0.integral_coefficients == set([f])
    assert id0.enabled_coefficients == [True, False]
    assert id1.domain is mesh2
    assert id1.integral_type == 'exterior_facet'
    assert id1.subdomain_id == (888, )
    assert fd.original_form.domain_numbering()[id1.domain] == 1
    assert id1.integral_coefficients == set([f, g])
    assert id1.enabled_coefficients == [True, True]


def test_mixed_function_space_with_mixed_mesh_restriction():
    cell = triangle
    elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1, (2, ), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2])
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=101)
    mesh2 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=102)
    domain = MixedMesh(mesh0, mesh1, mesh2)
    V = FunctionSpace(domain, elem)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    V2 = FunctionSpace(mesh2, elem2)
    u1 = TrialFunction(V1)
    v2 = TestFunction(V2)
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    f0, f1, f2 = split(f)
    g0, g1, g2 = split(g)
    dS1 = Measure("dS", mesh1)
    x2 = SpatialCoordinate(mesh2)
    form = inner(x2, g1) * g2 * inner(u1('-'), grad(v2('|'))) * dS1(999)
    fd = compute_form_data(form,
                           do_apply_function_pullbacks=True,
                           do_apply_integral_scaling=True,
                           do_apply_geometry_lowering=True,
                           preserve_geometry_types=(CellVolume, FacetArea),
                           do_apply_restrictions=True,
                           do_estimate_degrees=True,
                           do_split_coefficients=(f, g),
                           do_assume_single_integral_type=False,
                           complex_mode=False)
    integral_data, = fd.integral_data
    assert integral_data.domain_integral_type_map[mesh1] == "interior_facet"
    assert integral_data.domain_integral_type_map[mesh2] == "exterior_facet"


def test_mixed_function_space_with_mixed_mesh_signature():
    cell = triangle
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pullback, H1), ufl_id=101)
    dx0 = Measure("dx", mesh0)
    dx1 = Measure("dx", mesh1)
    n0 = FacetNormal(mesh0)
    n1 = FacetNormal(mesh1)
    form_a = inner(n1, n1) * dx0(999)
    form_b = inner(n0, n0) * dx1(999)
    assert form_a.signature() == form_b.signature()
    assert extract_domains(form_a) == (mesh0, mesh1)
    assert extract_domains(form_b) == (mesh1, mesh0)
