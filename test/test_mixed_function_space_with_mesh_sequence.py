import pytest
from utils import FiniteElement, LagrangeElement, MixedElement

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
    derivative,
    div,
    dot,
    grad,
    inner,
    split,
    tetrahedron,
    triangle,
)
from ufl.algorithms import compute_form_data
from ufl.cell import Cell, CellSequence
from ufl.domain import extract_domains
from ufl.pullback import contravariant_piola, identity_pullback
from ufl.sobolevspace import H1, L2, HDiv


def test_mixed_function_space_with_mesh_sequence_cell():
    cell = triangle
    elem0 = LagrangeElement(cell, 1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 2, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 1, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2], make_cell_sequence=True)
    mesh0 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=100)
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=101)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=102)
    domain = MeshSequence([mesh0, mesh1, mesh2])
    V = FunctionSpace(domain, elem)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    u1 = TrialFunction(V1)
    v0 = TestFunction(V0)
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    f0, _f1, _f2 = split(f)
    _g0, g1, _g2 = split(g)
    dx2 = Measure(
        "dx",
        mesh2,
        intersect_measures=(
            Measure("dx", mesh0),
            Measure("dx", mesh1),
        ),
    )
    x1 = SpatialCoordinate(mesh1)
    # Assemble (0, 1)-block.
    form = x1[1] * f0 * div(g1) * inner(u1, grad(v0)) * dx2(999)
    fd = compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=True,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(CellVolume, FacetArea),
        do_apply_restrictions=True,
        do_estimate_degrees=True,
        do_replace_functions=True,
        coefficients_to_split=(f, g),
        complex_mode=False,
    )
    (id0,) = fd.integral_data
    assert fd.preprocessed_form.arguments() == (v0, u1)
    assert fd.reduced_coefficients == [f, g]
    assert form.coefficients()[fd.original_coefficient_positions[0]] is f
    assert form.coefficients()[fd.original_coefficient_positions[1]] is g
    assert id0.domain_integral_type_map[mesh0] == "cell"
    assert id0.domain_integral_type_map[mesh1] == "cell"
    assert id0.domain_integral_type_map[mesh2] == "cell"
    assert id0.domain is mesh2
    assert id0.integral_type == "cell"
    assert id0.subdomain_id == (999,)
    assert fd.original_form.domain_numbering()[id0.domain] == 0
    assert id0.integral_coefficients == set([f, g])
    assert id0.enabled_coefficients == [True, True]


def test_mixed_function_space_with_mesh_sequence_facet():
    cell = triangle
    elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 2, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 1, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2], make_cell_sequence=True)
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=101)
    mesh2 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=102)
    domain = MeshSequence([mesh0, mesh1, mesh2])
    V = FunctionSpace(domain, elem)
    V1 = FunctionSpace(mesh1, elem1)
    V2 = FunctionSpace(mesh2, elem2)
    u1 = TrialFunction(V1)
    v2 = TestFunction(V2)
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    f0, f1, _f2 = split(f)
    _g0, g1, g2 = split(g)
    dS1 = Measure(
        "dS",
        mesh1,
        intersect_measures=(Measure("ds", mesh2),),
    )
    ds2 = Measure(
        "ds",
        mesh2,
        intersect_measures=(
            Measure("dS", mesh0),
            Measure("ds", mesh1),
        ),
    )
    x2 = SpatialCoordinate(mesh2)
    # Assemble (2, 1)-block.
    form = inner(x2, g1("+")) * g2 * inner(u1("-"), grad(v2)) * dS1(999) + f0("-") * div(
        f1
    ) * inner(div(u1), v2) * ds2(777)
    fd = compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=True,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(CellVolume, FacetArea),
        do_apply_restrictions=True,
        do_estimate_degrees=True,
        do_replace_functions=True,
        coefficients_to_split=(f, g),
        complex_mode=False,
    )
    (
        id0,
        id1,
    ) = fd.integral_data
    assert fd.preprocessed_form.arguments() == (v2, u1)
    assert fd.reduced_coefficients == [f, g]
    assert form.coefficients()[fd.original_coefficient_positions[0]] is f
    assert form.coefficients()[fd.original_coefficient_positions[1]] is g
    assert id0.domain_integral_type_map[mesh1] == "interior_facet"
    assert id0.domain_integral_type_map[mesh2] == "exterior_facet"
    assert id0.domain is mesh1
    assert id0.integral_type == "interior_facet"
    assert id0.subdomain_id == (999,)
    assert fd.original_form.domain_numbering()[id0.domain] == 0
    assert id0.integral_coefficients == set([g])
    assert id0.enabled_coefficients == [False, True]
    assert id1.domain_integral_type_map[mesh0] == "interior_facet"
    assert id1.domain_integral_type_map[mesh1] == "exterior_facet"
    assert id1.domain_integral_type_map[mesh2] == "exterior_facet"
    assert id1.domain is mesh2
    assert id1.integral_type == "exterior_facet"
    assert id1.subdomain_id == (777,)
    assert fd.original_form.domain_numbering()[id1.domain] == 1
    assert id1.integral_coefficients == set([f])
    assert id1.enabled_coefficients == [True, False]


def test_mixed_function_space_with_mesh_sequence_signature():
    cell = triangle
    mesh0 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=100)
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=101)
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
    elem0 = LagrangeElement(cell, 1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2], make_cell_sequence=True)
    mesh0 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=100)
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=101)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=102)
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


def test_mixed_function_space_with_mesh_sequence_raise():
    cell = triangle
    elem0 = FiniteElement("Lagrange", cell, 1, (), identity_pullback, H1)
    elem1 = FiniteElement("Brezzi-Douglas-Marini", cell, 1, (2,), contravariant_piola, HDiv)
    elem2 = FiniteElement("Discontinuous Lagrange", cell, 0, (), identity_pullback, L2)
    elem = MixedElement([elem0, elem1, elem2], make_cell_sequence=True)
    mesh0 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=101)
    mesh2 = Mesh(FiniteElement("Lagrange", cell, 1, (2,), identity_pullback, H1), ufl_id=102)
    domain = MeshSequence([mesh0, mesh1, mesh2])
    V = FunctionSpace(domain, elem)
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    _, f1, _ = split(f)
    _, g1, _ = split(g)
    dS1 = Measure("dS", mesh1)
    # Make sure that all mixed functions are split when applying default restrictions.
    form = div(g1("+")) * div(f1("-")) * dS1
    with pytest.raises(ValueError) as e_info:
        _ = compute_form_data(
            form,
            do_apply_function_pullbacks=True,
            do_apply_integral_scaling=True,
            do_apply_geometry_lowering=True,
            preserve_geometry_types=(CellVolume, FacetArea),
            do_apply_restrictions=True,
            do_estimate_degrees=True,
            do_replace_functions=True,
            coefficients_to_split=(f,),
            complex_mode=False,
        )
    assert e_info.match("Found multiple domains, cannot return just one.")
    # Make sure that g1 is restricted as f1.
    form = div(g1) * div(f1("-")) * dS1
    with pytest.raises(ValueError) as e_info:
        _ = compute_form_data(
            form,
            do_apply_function_pullbacks=True,
            do_apply_integral_scaling=True,
            do_apply_geometry_lowering=True,
            preserve_geometry_types=(CellVolume, FacetArea),
            do_apply_restrictions=True,
            do_estimate_degrees=True,
            do_replace_functions=True,
            coefficients_to_split=(f, g),
            complex_mode=False,
        )
    assert e_info.match("Discontinuous type Coefficient must be restricted.")


def test_mixed_function_space_with_mesh_sequence_quad_triangle():
    dim = 2
    coord_degree = 1
    cell_q = Cell("quadrilateral")
    cell_t = Cell("triangle")
    mesh_q = Mesh(LagrangeElement(cell_q, coord_degree, (dim,)))
    mesh_t = Mesh(LagrangeElement(cell_t, coord_degree, (dim,)))
    mesh = MeshSequence([mesh_q, mesh_t])
    assert mesh.ufl_cell() == CellSequence([cell_q, cell_t])
    elem_q = LagrangeElement(cell_q, 1, ())  # Q1
    elem_t = LagrangeElement(cell_t, 1, ())  # P1
    elem = MixedElement([elem_q, elem_t], make_cell_sequence=True)
    assert elem.cell == CellSequence([cell_q, cell_t])
    dx_q = Measure("dx", mesh_q)
    dx_t = Measure("dx", mesh_t)
    ds_q = Measure(
        "ds",
        mesh_q,
        intersect_measures=(Measure("ds", mesh_t),),
    )
    ds_t = Measure(
        "ds",
        mesh_t,
        intersect_measures=(Measure("ds", mesh_q),),
    )
    V = FunctionSpace(mesh, elem)
    u = Coefficient(V)
    v = TestFunction(V)
    u_q, u_t = split(u)
    v_q, v_t = split(v)
    n_q = FacetNormal(mesh_q)
    n_t = FacetNormal(mesh_t)
    C = 100.0
    h = 0.1  # mesh size
    interface_id = 999  # subdomain_id for the interface
    F = (
        inner(grad(u_q), grad(v_q)) * dx_q
        + inner(grad(u_t), grad(v_t)) * dx_t
        - inner((grad(u_q) + grad(u_t)) / 2, (v_q * n_q + v_t * n_t)) * ds_q(interface_id)
        - inner((u_q * n_q + u_t * n_t), (grad(v_q) + grad(v_t)) / 2) * ds_t(interface_id)
        + C / h * inner(u_q - u_t, v_q - v_t) * ds_q(interface_id)
    )
    _ = derivative(F, u)


def test_mixed_function_space_with_mesh_sequence_tetrahedron_triangle():
    mesh0 = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1), ufl_id=100)
    mesh1 = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3,), identity_pullback, H1), ufl_id=101)
    mesh2 = Mesh(FiniteElement("Lagrange", triangle, 1, (3,), identity_pullback, H1), ufl_id=102)
    mesh = MeshSequence([mesh0, mesh1, mesh2])
    assert mesh.ufl_cell() == CellSequence([tetrahedron, tetrahedron, triangle])
    elem0 = FiniteElement("Brezzi-Douglas-Marini", tetrahedron, 1, (3,), contravariant_piola, HDiv)
    elem1 = FiniteElement("Discontinuous Lagrange", tetrahedron, 0, (), identity_pullback, L2)
    elem2 = FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1)
    elem = MixedElement([elem0, elem1, elem2], make_cell_sequence=True)
    assert elem.cell == CellSequence([tetrahedron, tetrahedron, triangle])
    V = FunctionSpace(mesh, elem)
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    u1 = TrialFunction(V1)
    v0 = TestFunction(V0)
    f = Coefficient(V, count=1000)
    _f0, _f1, f2 = split(f)
    n0 = FacetNormal(mesh0)
    # Assemble (0, 1)-block.
    dx2_ds0_dS1 = Measure(
        "dx",
        mesh2,
        intersect_measures=(
            Measure("ds", mesh0),
            Measure("dS", mesh1),
        ),
    )
    form = inner(grad(f2)[2] * u1("+"), dot(v0, n0)) * dx2_ds0_dS1(999)
    fd = compute_form_data(
        form,
        do_apply_function_pullbacks=True,
        do_apply_integral_scaling=True,
        do_apply_geometry_lowering=True,
        preserve_geometry_types=(CellVolume, FacetArea),
        do_apply_restrictions=True,
        do_estimate_degrees=True,
        do_replace_functions=True,
        coefficients_to_split=(f,),
        complex_mode=False,
    )
    (id0,) = fd.integral_data
    assert fd.preprocessed_form.arguments() == (v0, u1)
    assert fd.reduced_coefficients == [
        f,
    ]
    assert form.coefficients()[fd.original_coefficient_positions[0]] is f
    assert id0.domain_integral_type_map[mesh0] == "exterior_facet"
    assert id0.domain_integral_type_map[mesh1] == "interior_facet"
    assert id0.domain_integral_type_map[mesh2] == "cell"
    assert id0.domain is mesh2
    assert id0.integral_type == "cell"
    assert id0.subdomain_id == (999,)
    assert fd.original_form.domain_numbering()[id0.domain] == 0
    assert id0.integral_coefficients == set([f])
    assert id0.enabled_coefficients == [True]
