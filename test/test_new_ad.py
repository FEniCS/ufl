from ufl import (CellVolume, Coefficient, Constant, FacetNormal, FiniteElement, FunctionSpace, Identity, Mesh,
                 SpatialCoordinate, TestFunction, VectorConstant, VectorElement, as_ufl, cos, derivative, diff, exp,
                 grad, ln, sin, tan, triangle, variable, zero)
from ufl.algorithms.apply_derivatives import GenericDerivativeRuleset, GradRuleset, apply_derivatives
from ufl.algorithms.renumbering import renumber_indices

# Note: the old tests in test_automatic_differentiation.py are a bit messy
#       but still cover many things that are not in here yet.


# FIXME: Write UNIT tests for all terminal derivatives!
# FIXME: Write UNIT tests for operator derivatives!


def test_apply_derivatives_doesnt_change_expression_without_derivatives():
    cell = triangle
    d = cell.geometric_dimension()
    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)

    domain = Mesh(VectorElement("Lagrange", cell, 1))
    v0_space = FunctionSpace(domain, V0)
    v1_space = FunctionSpace(domain, V1)

    # Literals
    z = zero((3, 2))
    one = as_ufl(1)
    two = as_ufl(2.0)
    ident = Identity(d)
    literals = [z, one, two, ident]

    # Geometry
    x = SpatialCoordinate(domain)
    n = FacetNormal(domain)
    volume = CellVolume(domain)
    geometry = [x, n, volume]

    # Arguments
    v0 = TestFunction(v0_space)
    v1 = TestFunction(v1_space)
    arguments = [v0, v1]

    # Coefficients
    f0 = Coefficient(v0_space)
    f1 = Coefficient(v1_space)
    coefficients = [f0, f1]

    # Expressions
    e0 = f0 + f1
    e1 = v0 * (f1/3 - f0**2)
    e2 = exp(sin(cos(tan(ln(x[0])))))
    expressions = [e0, e1, e2]

    # Check that all are unchanged
    for expr in literals + geometry + arguments + coefficients + expressions:
        # Note the use of "is" here instead of ==, this property
        # is important for efficiency and memory usage
        assert apply_derivatives(expr) is expr


def test_literal_derivatives_are_zero():
    cell = triangle
    d = cell.geometric_dimension()

    # Literals
    one = as_ufl(1)
    two = as_ufl(2.0)
    ident = Identity(d)
    literals = [one, two, ident]

    # Generic ruleset handles literals directly:
    for lit in literals:
        for sh in [(), (d,), (d, d+1)]:
            assert GenericDerivativeRuleset(sh)(lit) == zero(lit.ufl_shape + sh)

    # Variables
    v0 = variable(one)
    v1 = variable(zero((d,)))
    v2 = variable(ident)
    variables = [v0, v1, v2]

    # Test literals via apply_derivatives and variable ruleset:
    for lit in literals:
        for v in variables:
            assert apply_derivatives(diff(lit, v)) == zero(lit.ufl_shape + v.ufl_shape)

    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)
    domain = Mesh(VectorElement("Lagrange", cell, 1))
    v0_space = FunctionSpace(domain, V0)
    v1_space = FunctionSpace(domain, V1)
    u0 = Coefficient(v0_space)
    u1 = Coefficient(v1_space)
    v0 = TestFunction(v0_space)
    v1 = TestFunction(v1_space)
    args = [(u0, v0), (u1, v1)]

    # Test literals via apply_derivatives and variable ruleset:
    for lit in literals:
        for u, v in args:
            assert apply_derivatives(derivative(lit, u, v)) == zero(lit.ufl_shape + v.ufl_shape)

    # Test grad ruleset directly since grad(literal) is invalid:
    assert GradRuleset(d)(one) == zero((d,))
    assert GradRuleset(d)(one) == zero((d,))


def test_grad_ruleset():
    cell = triangle
    d = cell.geometric_dimension()

    V0 = FiniteElement("DG", cell, 0)
    V1 = FiniteElement("Lagrange", cell, 1)
    V2 = FiniteElement("Lagrange", cell, 2)
    W0 = VectorElement("DG", cell, 0)
    W1 = VectorElement("Lagrange", cell, 1)
    W2 = VectorElement("Lagrange", cell, 2)

    domain = Mesh(VectorElement("Lagrange", cell, 1))
    v0_space = FunctionSpace(domain, V0)
    v1_space = FunctionSpace(domain, V1)
    v2_space = FunctionSpace(domain, V2)
    w0_space = FunctionSpace(domain, W0)
    w1_space = FunctionSpace(domain, W1)
    w2_space = FunctionSpace(domain, W2)

    # Literals
    one = as_ufl(1)
    two = as_ufl(2.0)
    ident = Identity(d)

    # Geometry
    x = SpatialCoordinate(domain)
    n = FacetNormal(domain)
    volume = CellVolume(domain)

    # Arguments
    u0 = TestFunction(v0_space)
    u1 = TestFunction(v1_space)
    arguments = [u0, u1]

    # Coefficients
    r = Constant(domain)
    vr = VectorConstant(domain)
    f0 = Coefficient(v0_space)
    f1 = Coefficient(v1_space)
    f2 = Coefficient(v2_space)
    vf0 = Coefficient(w0_space)
    vf1 = Coefficient(w1_space)
    vf2 = Coefficient(w2_space)

    rules = GradRuleset(d)

    # Literals
    assert rules(one) == zero((d,))
    assert rules(two) == zero((d,))
    assert rules(ident) == zero((d, d, d))

    # Assumed piecewise constant geometry
    for g in [n, volume]:
        assert rules(g) == zero(g.ufl_shape + (d,))

    # Non-constant geometry
    assert rules(x) == ident

    # Arguments
    for u in arguments:
        assert rules(u) == grad(u)

    # Piecewise constant coefficients (Constant)
    assert rules(r) == zero((d,))
    assert rules(vr) == zero((d, d))
    assert rules(grad(r)) == zero((d, d))
    assert rules(grad(vr)) == zero((d, d, d))

    # Piecewise constant coefficients (DG0)
    assert rules(f0) == zero((d,))
    assert rules(vf0) == zero((d, d))
    assert rules(grad(f0)) == zero((d, d))
    assert rules(grad(vf0)) == zero((d, d, d))

    # Piecewise linear coefficients
    assert rules(f1) == grad(f1)
    assert rules(vf1) == grad(vf1)
    # assert rules(grad(f1)) == zero((d,d)) # TODO: Use degree to make this work
    # assert rules(grad(vf1)) == zero((d,d,d))

    # Piecewise quadratic coefficients
    assert rules(grad(f2)) == grad(grad(f2))
    assert rules(grad(vf2)) == grad(grad(vf2))

    # Indexed coefficients
    assert renumber_indices(apply_derivatives(grad(vf2[0]))) == renumber_indices(grad(vf2)[0, :])
    assert renumber_indices(apply_derivatives(grad(vf2[1])[0])) == renumber_indices(grad(vf2)[1, 0])

    # Grad of gradually more complex expressions
    assert apply_derivatives(grad(2*f0)) == zero((d,))
    assert renumber_indices(apply_derivatives(grad(2*f1))) == renumber_indices(2*grad(f1))
    assert renumber_indices(apply_derivatives(grad(sin(f1)))) == renumber_indices(cos(f1) * grad(f1))
    assert renumber_indices(apply_derivatives(grad(cos(f1)))) == renumber_indices(-sin(f1) * grad(f1))


def test_variable_ruleset():
    pass


def test_gateaux_ruleset():
    pass
