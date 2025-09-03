import pytest
from utils import FiniteElement, LagrangeElement, MixedElement

from ufl import (
    Action,
    Adjoint,
    Coefficient,
    Constant,
    FacetNormal,
    FunctionSpace,
    Interpolate,
    Matrix,
    Measure,
    Mesh,
    MeshSequence,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    cos,
    div,
    grad,
    inner,
    split,
    triangle,
)
from ufl.domain import extract_unique_domain
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
        assert extract_unique_domain(u_i) == domain[i]

    f = Coefficient(V)
    f1, f2, f3 = split(f)
    for i, f_i in enumerate((f1, f2, f3)):
        assert extract_unique_domain(f_i) == domain[i]

    x1, y1 = SpatialCoordinate(mesh1)
    expr = u1 + x1 * cos(x1)
    assert extract_unique_domain(expr) == mesh1

    expr2 = u1 * Constant(mesh1) + x1
    assert extract_unique_domain(expr2) == mesh1

    x2, y2 = SpatialCoordinate(mesh2)
    with pytest.raises(ValueError) as e_info:
        _ = extract_unique_domain(u1 + u2)
        _ = extract_unique_domain(u1 + u2 + x2 * cos(x2 * u1))


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
    v = TestFunction(V)
    v1, v2, v3 = split(v)

    f = Coefficient(V)
    f1, f2, f3 = split(f)

    n = FacetNormal(mesh1)
    dx1 = Measure("dx", mesh1)
    ds1 = Measure("ds", mesh1)
    dx2 = Measure("dx", mesh2)

    form1 = inner(grad(u1), grad(v1)) * dx1 - inner(grad(u1), n) * v1 * ds1
    assert extract_unique_domain(form1) == mesh1

    form2 = inner(u1, f1) * dx1
    assert extract_unique_domain(form2) == mesh1

    form3 = inner(u1, v1) * dx1 + inner(u2, v2) * dx2
    with pytest.raises(ValueError):
        extract_unique_domain(form3)


def test_extract_unique_domain_single_mesh():
    """Test domain extraction for standard function spaces on a single mesh."""
    cell = triangle
    mesh = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=200)

    # Test scalar elements
    P1 = LagrangeElement(cell, 1)
    V_scalar = FunctionSpace(mesh, P1)
    u_scalar = TrialFunction(V_scalar)
    f_scalar = Coefficient(V_scalar)

    assert extract_unique_domain(u_scalar) == mesh
    assert extract_unique_domain(f_scalar) == mesh

    P1_vec = LagrangeElement(cell, 1, (2,))
    V_vector = FunctionSpace(mesh, P1_vec)
    u_vector = TrialFunction(V_vector)
    f_vector = Coefficient(V_vector)

    assert extract_unique_domain(u_vector) == mesh
    assert extract_unique_domain(f_vector) == mesh

    assert extract_unique_domain(u_vector[0]) == mesh
    assert extract_unique_domain(u_vector[1]) == mesh
    assert extract_unique_domain(f_vector[0]) == mesh
    assert extract_unique_domain(f_vector[1]) == mesh

    P1_tensor = LagrangeElement(cell, 1, (2, 2))
    V_tensor = FunctionSpace(mesh, P1_tensor)
    u_tensor = TrialFunction(V_tensor)
    f_tensor = Coefficient(V_tensor)

    assert extract_unique_domain(u_tensor) == mesh
    assert extract_unique_domain(f_tensor) == mesh
    assert extract_unique_domain(u_tensor[0, 0]) == mesh
    assert extract_unique_domain(u_tensor[1, 1]) == mesh
    assert extract_unique_domain(f_tensor[0, 1]) == mesh

    x, y = SpatialCoordinate(mesh)
    expr1 = u_scalar + f_scalar
    expr2 = u_vector[0] + x
    expr3 = inner(u_vector, f_vector)

    assert extract_unique_domain(expr1) == mesh
    assert extract_unique_domain(expr2) == mesh
    assert extract_unique_domain(expr3) == mesh

    # Test forms
    dx = Measure("dx", mesh)
    form = inner(u_scalar, f_scalar) * dx
    assert extract_unique_domain(form) == mesh


def test_extract_unique_domain_mixed_scalar_vector_tensor():
    """Test domain extraction for mixed function spaces
    with scalar, vector, and tensor elements."""
    cell = triangle
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=400)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=401)
    mesh3 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=402)
    domain = MeshSequence([mesh1, mesh2, mesh3])

    scalar_elem = LagrangeElement(cell, 1)
    vector_elem = LagrangeElement(cell, 1, (2,))
    tensor_elem = LagrangeElement(cell, 1, (2, 2))
    mixed_elem = MixedElement([scalar_elem, vector_elem, tensor_elem])

    V = FunctionSpace(domain, mixed_elem)
    u = TrialFunction(V)
    f = Coefficient(V)

    u_scalar, u_vector, u_tensor = split(u)
    f_scalar, f_vector, f_tensor = split(f)

    for i, u_i in enumerate((u_scalar, u_vector, u_tensor)):
        assert extract_unique_domain(u_i) == domain[i]
    for i, f_i in enumerate((f_scalar, f_vector, f_tensor)):
        assert extract_unique_domain(f_i) == domain[i]

    for i in range(2):
        assert extract_unique_domain(u_vector[i]) == mesh2
        assert extract_unique_domain(f_vector[i]) == mesh2

    for i in range(2):
        for j in range(2):
            assert extract_unique_domain(u_tensor[i, j]) == mesh3
            assert extract_unique_domain(f_tensor[i, j]) == mesh3

    x1, y1 = SpatialCoordinate(mesh1)
    x2, y2 = SpatialCoordinate(mesh2)
    x3, y3 = SpatialCoordinate(mesh3)

    expr_scalar = u_scalar * y1 + f_scalar + x1
    assert extract_unique_domain(expr_scalar) == mesh1

    expr_vector = inner(u_vector * y2, f_vector) + x2
    assert extract_unique_domain(expr_vector) == mesh2

    expr_vec_comp = u_vector[0] + f_vector[1] * y2 + x2
    assert extract_unique_domain(expr_vec_comp) == mesh2

    expr_tensor = y3 * u_tensor[0, 0] + f_tensor[1, 1] + x3
    assert extract_unique_domain(expr_tensor) == mesh3

    with pytest.raises(ValueError):
        extract_unique_domain(u_scalar + u_vector[0])

    with pytest.raises(ValueError):
        extract_unique_domain(u_vector[0] + u_tensor[0, 0])

    with pytest.raises(ValueError):
        extract_unique_domain(f_scalar + f_tensor[1, 1])

    with pytest.raises(ValueError):
        extract_unique_domain(u_scalar + x2)

    with pytest.raises(ValueError):
        extract_unique_domain(u_vector[0] + x3)

    dx1 = Measure("dx", mesh1)
    dx2 = Measure("dx", mesh2)
    dx3 = Measure("dx", mesh3)

    form_scalar = u_scalar * f_scalar * dx1
    form_vector = inner(u_vector, f_vector) * dx2
    form_tensor = u_tensor[0, 0] * f_tensor[1, 1] * dx3

    assert extract_unique_domain(form_scalar) == mesh1
    assert extract_unique_domain(form_vector) == mesh2
    assert extract_unique_domain(form_tensor) == mesh3

    div_expr = div(u_vector) * f_scalar
    with pytest.raises(ValueError):
        extract_unique_domain(div_expr)


def test_extract_unique_domain_repeated_meshes():
    cell = triangle
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=500)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=501)

    # MeshSequence with repeated meshes
    domain_repeated = MeshSequence([mesh1, mesh2, mesh1])

    scalar_elem = LagrangeElement(cell, 1, shape=())
    mixed_elem = MixedElement([scalar_elem, scalar_elem, scalar_elem])
    V = FunctionSpace(domain_repeated, mixed_elem)
    u = TrialFunction(V)

    u1, u2, u3 = split(u)

    assert extract_unique_domain(u1) == mesh1
    assert extract_unique_domain(u2) == mesh2
    assert extract_unique_domain(u3) == mesh1

    expr_same = u1 + u3
    assert extract_unique_domain(expr_same) == mesh1

    with pytest.raises(ValueError):
        extract_unique_domain(u1 + u2)


def test_extract_unique_domain_baseform():
    cell = triangle
    mesh1 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=400)
    mesh2 = Mesh(LagrangeElement(cell, 1, (2,)), ufl_id=401)
    scalar_elem = LagrangeElement(cell, 1)

    V1 = FunctionSpace(mesh1, scalar_elem)
    V2 = FunctionSpace(mesh2, scalar_elem)

    A = Matrix(V1, V2)
    assert extract_unique_domain(A) == mesh1

    v = Coefficient(V2)
    action_Au = Action(A, v)
    assert extract_unique_domain(action_Au) == mesh1

    Astar = Adjoint(A)
    assert extract_unique_domain(Astar) == mesh2

    v1 = TrialFunction(V1)
    v2star = TestFunction(V2.dual())
    interp = Interpolate(v1, v2star)  # V1 x V2^* -> R, equiv V1 -> V2
    assert extract_unique_domain(interp) == mesh2
    adjoint_interp = Adjoint(interp)  # V2^* x V1 -> R, equiv V2^* -> V1^*
    assert extract_unique_domain(adjoint_interp) == mesh1

    cofunc = Coefficient(V2.dual())
    scalar = Action(cofunc, v)
    assert extract_unique_domain(scalar) is None

    v = TestFunction(V2)
    dx = Measure("dx", mesh2)
    one_form = v * dx
    formsum = cofunc + one_form
    assert extract_unique_domain(formsum) is mesh2

    two_form = interp * v * dx
    assert extract_unique_domain(two_form) is mesh2
