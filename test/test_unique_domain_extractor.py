from utils import FiniteElement, LagrangeElement, MixedElement

import pytest
from ufl import (
    Coefficient,
    FunctionSpace,
    Mesh,
    MeshSequence,
    SpatialCoordinate,
    Measure,
    Argument,
    Interpolate,
    TrialFunction,
    split,
    triangle,
    cos,
    inner,
    div
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


def test_extract_unique_domain_single_mesh():
    """Test domain extraction for standard function spaces on a single mesh."""
    cell = triangle
    mesh = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=200)
    
    # Test scalar elements
    P1 = LagrangeElement(cell, 1)
    V_scalar = FunctionSpace(mesh, P1)
    u_scalar = TrialFunction(V_scalar)
    f_scalar = Coefficient(V_scalar)
    
    # Test that single mesh functions return the correct domain
    assert extract_unique_domain_dag(u_scalar) == mesh
    assert extract_unique_domain_dag(f_scalar) == mesh
    
    # Test vector elements
    P1_vec = LagrangeElement(cell, 1, (2,))
    V_vector = FunctionSpace(mesh, P1_vec)
    u_vector = TrialFunction(V_vector)
    f_vector = Coefficient(V_vector)
    
    assert extract_unique_domain_dag(u_vector) == mesh
    assert extract_unique_domain_dag(f_vector) == mesh
    
    # Test indexing into vector elements
    assert extract_unique_domain_dag(u_vector[0]) == mesh
    assert extract_unique_domain_dag(u_vector[1]) == mesh
    assert extract_unique_domain_dag(f_vector[0]) == mesh
    assert extract_unique_domain_dag(f_vector[1]) == mesh
    
    # Test tensor elements
    P1_tensor = LagrangeElement(cell, 1, (2, 2))
    V_tensor = FunctionSpace(mesh, P1_tensor)
    u_tensor = TrialFunction(V_tensor)
    f_tensor = Coefficient(V_tensor)
    
    assert extract_unique_domain_dag(u_tensor) == mesh
    assert extract_unique_domain_dag(f_tensor) == mesh
    assert extract_unique_domain_dag(u_tensor[0, 0]) == mesh
    assert extract_unique_domain_dag(u_tensor[1, 1]) == mesh
    assert extract_unique_domain_dag(f_tensor[0, 1]) == mesh
    
    # Test expressions combining same-domain functions
    x, y = SpatialCoordinate(mesh)
    expr1 = u_scalar + f_scalar
    expr2 = u_vector[0] + x
    expr3 = inner(u_vector, f_vector)
    
    assert extract_unique_domain_dag(expr1) == mesh
    assert extract_unique_domain_dag(expr2) == mesh
    assert extract_unique_domain_dag(expr3) == mesh
    
    # Test forms
    dx = Measure("dx", mesh)
    form = inner(u_scalar, f_scalar) * dx
    assert extract_unique_domain_dag(form) == mesh


def test_extract_unique_domain_mixed_scalar_vector_tensor():
    """Test domain extraction for mixed function spaces with scalar, vector, and tensor elements."""
    cell = triangle
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=400)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=401)
    mesh3 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=402)
    domain = MeshSequence([mesh1, mesh2, mesh3])
    
    # Create mixed element with different types:
    # - scalar on mesh1
    # - vector on mesh2 
    # - tensor on mesh3
    scalar_elem = LagrangeElement(cell, 1)                   # 1 component
    vector_elem = LagrangeElement(cell, 1, (2,))             # 2 components
    tensor_elem = LagrangeElement(cell, 1, (2, 2))           # 4 components
    mixed_elem = MixedElement([scalar_elem, vector_elem, tensor_elem])
    
    V = FunctionSpace(domain, mixed_elem)
    u = TrialFunction(V)
    f = Coefficient(V)
    
    # For mixed element [scalar, vector(2), tensor(2,2)], split gives:
    # - u_components[0]: scalar (mesh1) - index 0
    # - u_components[1]: vector (mesh2) - indices 1-2  
    # - u_components[2]: tensor (mesh3) - indices 3-6
    u_scalar, u_vector, u_tensor = split(u)
    f_scalar, f_vector, f_tensor = split(f)
    
    # Test that each component maps to correct mesh
    for i, u_i in enumerate((u_scalar, u_vector, u_tensor)):
        assert extract_unique_domain_dag(u_i) == domain[i]
    for i, f_i in enumerate((f_scalar, f_vector, f_tensor)):
        assert extract_unique_domain_dag(f_i) == domain[i]
    
    # Test indexing into vector and tensor components
    for i in range(2):
        assert extract_unique_domain_dag(u_vector[i]) == mesh2
        assert extract_unique_domain_dag(f_vector[i]) == mesh2
    
    for i in range(2):
        for j in range(2):
            assert extract_unique_domain_dag(u_tensor[i, j]) == mesh3
            assert extract_unique_domain_dag(f_tensor[i, j]) == mesh3
    
    # Test expressions on same mesh (should work)
    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    x3, y3 = SpatialCoordinate(mesh3)
    
    # Scalar expressions
    expr_scalar = u_scalar * y1 + f_scalar + x1
    assert extract_unique_domain_dag(expr_scalar) == mesh1
    
    # Vector expressions
    expr_vector = inner(u_vector * y2, f_vector) + x2
    assert extract_unique_domain_dag(expr_vector) == mesh2
    
    # Vector component expressions
    expr_vec_comp = u_vector[0] + f_vector[1] * y2 + x2
    assert extract_unique_domain_dag(expr_vec_comp) == mesh2
    
    # Tensor expressions
    expr_tensor = y3 * u_tensor[0, 0] + f_tensor[1, 1] + x3
    assert extract_unique_domain_dag(expr_tensor) == mesh3
    
    # Test expressions mixing different mesh components (should fail)
    with pytest.raises(ValueError, match="Cannot extract unique domain"):
        extract_unique_domain_dag(u_scalar + u_vector[0])
    
    with pytest.raises(ValueError, match="Cannot extract unique domain"):
        extract_unique_domain_dag(u_vector[0] + u_tensor[0, 0])
    
    with pytest.raises(ValueError, match="Cannot extract unique domain"):
        extract_unique_domain_dag(f_scalar + f_tensor[1, 1])
    
    with pytest.raises(ValueError, match="Cannot extract unique domain"):
        extract_unique_domain_dag(u_scalar + x2)
    
    with pytest.raises(ValueError, match="Cannot extract unique domain"):
        extract_unique_domain_dag(u_vector[0] + x3)
    
    # Test forms on individual meshes
    dx1 = Measure("dx", mesh1)
    dx2 = Measure("dx", mesh2)
    dx3 = Measure("dx", mesh3)
    
    form_scalar = u_scalar * f_scalar * dx1
    form_vector = inner(u_vector, f_vector) * dx2
    form_tensor = u_tensor[0, 0] * f_tensor[1, 1] * dx3
    
    assert extract_unique_domain_dag(form_scalar) == mesh1
    assert extract_unique_domain_dag(form_vector) == mesh2
    assert extract_unique_domain_dag(form_tensor) == mesh3
    
    div_expr = div(u_vector) * f_scalar  # Cross-mesh, should fail
    with pytest.raises(ValueError, match="Cannot extract unique domain"):
        extract_unique_domain_dag(div_expr)


def test_extract_unique_domain_repeated_meshes():
    """Test edge cases for domain extraction."""
    cell = triangle
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=500)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=501)
    
    # MeshSequence with repeated meshes
    domain_repeated = MeshSequence([mesh1, mesh2, mesh1])  # mesh1 appears twice
    
    scalar_elem = LagrangeElement(cell, 1, shape=())
    mixed_elem = MixedElement([scalar_elem, scalar_elem, scalar_elem])
    V = FunctionSpace(domain_repeated, mixed_elem)
    u = TrialFunction(V)
    
    u1, u2, u3 = split(u)
    
    # Components 0 and 2 should map to mesh1, component 1 to mesh2
    assert extract_unique_domain_dag(u1) == mesh1  # index 0 -> mesh1
    assert extract_unique_domain_dag(u2) == mesh2  # index 1 -> mesh2
    assert extract_unique_domain_dag(u3) == mesh1  # index 2 -> mesh1 (repeated)
    
    # Expressions combining components on same underlying mesh should work
    expr_same = u1 + u3  # Both on mesh1
    assert extract_unique_domain_dag(expr_same) == mesh1
    
    # Expressions combining components on different meshes should fail
    with pytest.raises(ValueError, match="Cannot extract unique domain"):
        extract_unique_domain_dag(u1 + u2)  # mesh1 + mesh2


def test_extract_unique_domain_interpolate():
    cell = triangle
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=400)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=401)
    mesh3 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=402)
    domain = MeshSequence([mesh1, mesh2, mesh3])
    scalar_elem = LagrangeElement(cell, 1)                   # 1 component
    vector_elem = LagrangeElement(cell, 1, (2,))             # 2 components
    tensor_elem = LagrangeElement(cell, 1, (2, 2))           # 4 components
    mixed_elem = MixedElement([scalar_elem, vector_elem, tensor_elem])
    V = FunctionSpace(domain, mixed_elem)

    u = TrialFunction(V)
    f = Coefficient(V)

    u1, u2, u3 = split(u)
    f1, f2, f3 = split(f)

    # Interpolate a function on the mixed space
    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    x3, y3 = SpatialCoordinate(mesh3)
    interp_expr = x1 + cos(u1) * y1
    coarg = Argument(V.dual(), 0)
    vstar = split(coarg)
    Iu1 = Interpolate(interp_expr, vstar[0])
    expr = Iu1 + f1 * y1
    assert extract_unique_domain_dag(expr) == mesh1