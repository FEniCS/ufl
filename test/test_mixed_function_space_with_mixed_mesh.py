import pytest

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
    div,
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


def test_mixed_function_space_with_mixed_mesh_cell():
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
    V0 = FunctionSpace(mesh0, elem0)
    V1 = FunctionSpace(mesh1, elem1)
    u1 = TrialFunction(V1)
    v0 = TestFunction(V0)
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    f0, f1, f2 = split(f)
    g0, g1, g2 = split(g)
    dx2 = Measure("dx", mesh2)
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
        do_split_coefficients=(f, g),
        do_assume_single_integral_type=False,
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


def test_mixed_function_space_with_mixed_mesh_facet():
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
    V1 = FunctionSpace(mesh1, elem1)
    V2 = FunctionSpace(mesh2, elem2)
    u1 = TrialFunction(V1)
    v2 = TestFunction(V2)
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    f0, f1, f2 = split(f)
    g0, g1, g2 = split(g)
    dS1 = Measure("dS", mesh1)
    ds2 = Measure("ds", mesh2)
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
        do_split_coefficients=(f, g),
        do_assume_single_integral_type=False,
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


def test_mixed_function_space_with_mixed_mesh_raise():
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
    f = Coefficient(V, count=1000)
    g = Coefficient(V, count=2000)
    _, f1, _ = split(f)
    _, g1, _ = split(g)
    dS1 = Measure("dS", mesh1)
    # Make sure that all mixed functions are split when applying default restrictions.
    form = div(g1("+")) * div(f1("-")) * dS1
    with pytest.raises(RuntimeError) as e_info:
        _ = compute_form_data(
            form,
            do_apply_function_pullbacks=True,
            do_apply_integral_scaling=True,
            do_apply_geometry_lowering=True,
            preserve_geometry_types=(CellVolume, FacetArea),
            do_apply_restrictions=True,
            do_estimate_degrees=True,
            do_split_coefficients=(f,),
            do_assume_single_integral_type=False,
            complex_mode=False,
        )
    assert e_info.match("Not expecting a terminal object on a mixed mesh at this stage")
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
            do_split_coefficients=(f, g),
            do_assume_single_integral_type=False,
            complex_mode=False,
        )
    assert e_info.match("Discontinuous type Coefficient must be restricted.")


def test_mixed_function_space_with_mixed_mesh_signature():
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
