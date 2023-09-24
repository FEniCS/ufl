__authors__ = "India Marsden"
__date__ = "2020-12-28"

import pytest

from ufl import (Action, Adjoint, Argument, Coargument, Coefficient, Cofunction, FormSum, FunctionSpace, Matrix, Mesh,
                 MixedFunctionSpace, TestFunction, TrialFunction, action, adjoint, derivative, dx, inner, interval,
                 tetrahedron, triangle)
from ufl.algorithms.ad import expand_derivatives
from ufl.constantvalue import Zero
from ufl.duals import is_dual, is_primal
from ufl.finiteelement import FiniteElement
from ufl.form import ZeroBaseForm
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1


def test_mixed_functionspace(self):
    # Domains
    domain_3d = Mesh(FiniteElement("Lagrange", tetrahedron, 1, (3, ), identity_pull_back, H1))
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    domain_1d = Mesh(FiniteElement("Lagrange", interval, 1, (1, ), identity_pull_back, H1))
    # Finite elements
    f_1d = FiniteElement("Lagrange", interval, 1, (), identity_pull_back, H1)
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    f_3d = FiniteElement("Lagrange", tetrahedron, 1, (), identity_pull_back, H1)
    # Function spaces
    V_3d = FunctionSpace(domain_3d, f_3d)
    V_2d = FunctionSpace(domain_2d, f_2d)
    V_1d = FunctionSpace(domain_1d, f_1d)

    # MixedFunctionSpace = V_3d x V_2d x V_1d
    V = MixedFunctionSpace(V_3d, V_2d, V_1d)
    # Check sub spaces
    assert is_primal(V_3d)
    assert is_primal(V_2d)
    assert is_primal(V_1d)
    assert is_primal(V)

    # Get dual of V_3
    V_dual = V_3d.dual()

    #  Test dual functions on MixedFunctionSpace = V_dual x V_2d x V_1d
    V = MixedFunctionSpace(V_dual, V_2d, V_1d)
    V_mixed_dual = MixedFunctionSpace(V_dual, V_2d.dual(), V_1d.dual())

    assert is_dual(V_dual)
    assert not is_dual(V)
    assert is_dual(V_mixed_dual)


def test_dual_coefficients():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    v = Coefficient(V, count=1)
    u = Coefficient(V_dual, count=1)
    w = Cofunction(V_dual)

    assert is_primal(v)
    assert not is_dual(v)

    assert is_dual(u)
    assert not is_primal(u)

    assert is_dual(w)
    assert not is_primal(w)

    with pytest.raises(ValueError):
        Cofunction(V)


def test_dual_arguments():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    v = Argument(V, 1)
    u = Argument(V_dual, 2)
    w = Coargument(V_dual, 3)

    assert is_primal(v)
    assert not is_dual(v)

    assert is_dual(u)
    assert not is_primal(u)

    assert is_dual(w)
    assert not is_primal(w)

    with pytest.raises(ValueError):
        Coargument(V, 4)


def test_addition():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    u = TrialFunction(V)
    v = TestFunction(V)

    # linear 1-form
    L = v * dx
    a = Cofunction(V_dual)
    res = L + a
    assert isinstance(res, FormSum)
    assert res

    L = u * v * dx
    a = Matrix(V, V)
    res = L + a
    assert isinstance(res, FormSum)
    assert res

    # Check BaseForm._add__ simplification
    res += ZeroBaseForm((v, u))
    assert res == a + L
    # Check Form._add__ simplification
    L += ZeroBaseForm((v,))
    assert L == u * v * dx
    # Check BaseForm._add__ simplification
    res = ZeroBaseForm((v, u))
    res += a
    assert res == a
    # Check __neg__
    res = L
    res -= ZeroBaseForm((v,))
    assert res == L


def test_scalar_mult():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    V_dual = V.dual()

    # linear 1-form
    a = Cofunction(V_dual)
    res = 2 * a
    assert isinstance(res, FormSum)
    assert res

    a = Matrix(V, V)
    res = 2 * a
    assert isinstance(res, FormSum)
    assert res


def test_adjoint():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    a = Matrix(V, V)

    adj = adjoint(a)
    res = 2 * adj
    assert isinstance(res, FormSum)
    assert res

    res = adjoint(2 * a)
    assert isinstance(res, FormSum)
    assert isinstance(res.components()[0], Adjoint)

    # Adjoint(Adjoint(.)) = Id
    assert adjoint(adj) == a


def test_action():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    domain_1d = Mesh(FiniteElement("Lagrange", interval, 1, (1, ), identity_pull_back, H1))
    f_1d = FiniteElement("Lagrange", interval, 1, (), identity_pull_back, H1)
    U = FunctionSpace(domain_1d, f_1d)

    a = Matrix(V, U)
    b = Matrix(V, U.dual())
    u = Coefficient(U)
    u_a = Argument(U, 0)
    v = Coefficient(V)
    ustar = Cofunction(U.dual())
    u_form = u_a * dx

    res = action(a, u)
    assert res
    assert len(res.arguments()) < len(a.arguments())
    assert isinstance(res, Action)

    repeat = action(res, v)
    assert repeat
    assert len(repeat.arguments()) < len(res.arguments())

    res = action(2 * a, u)
    assert isinstance(res, FormSum)
    assert isinstance(res.components()[0], Action)

    res = action(b, u_form)
    assert res
    assert len(res.arguments()) < len(b.arguments())

    with pytest.raises(TypeError):
        res = action(a, v)

    with pytest.raises(TypeError):
        res = action(a, ustar)

    b2 = Matrix(V, U.dual())
    ustar2 = Cofunction(U.dual())
    # Check Action left-distributivity with FormSum
    res = action(b, ustar + ustar2)
    assert res == Action(b, ustar) + Action(b, ustar2)
    # Check Action right-distributivity with FormSum
    res = action(b + b2, ustar)
    assert res == Action(b, ustar) + Action(b2, ustar)

    a2 = Matrix(V, U)
    u2 = Coefficient(U)
    u3 = Coefficient(U)
    # Check Action left-distributivity with Sum
    # Add 3 Coefficients to check composition of Sum works fine since u + u2 + u3 => Sum(u, Sum(u2, u3))
    res = action(a, u + u2 + u3)
    assert res == Action(a, u3) + Action(a, u) + Action(a, u2)
    # Check Action right-distributivity with Sum
    res = action(a + a2, u)
    assert res == Action(a, u) + Action(a2, u)


def test_differentiation():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    domain_1d = Mesh(FiniteElement("Lagrange", interval, 1, (1, ), identity_pull_back, H1))
    f_1d = FiniteElement("Lagrange", interval, 1, (), identity_pull_back, H1)
    U = FunctionSpace(domain_1d, f_1d)

    u = Coefficient(U)
    v = Argument(U, 0)
    vstar = Argument(U.dual(), 0)

    # -- Cofunction -- #
    w = Cofunction(U.dual())
    dwdu = expand_derivatives(derivative(w, u))
    assert isinstance(dwdu, ZeroBaseForm)
    assert dwdu.arguments() == (Argument(w.ufl_function_space().dual(), 0), Argument(u.ufl_function_space(), 1))
    # Check compatibility with int/float
    assert dwdu == 0

    dwdw = expand_derivatives(derivative(w, w, vstar))
    assert dwdw == vstar

    dudw = expand_derivatives(derivative(u, w))
    # du/dw is a ufl.Zero and not a ZeroBaseForm
    # as we are not differentiating a BaseForm
    assert isinstance(dudw, Zero)
    assert dudw == 0

    # -- Coargument -- #
    dvstardu = expand_derivatives(derivative(vstar, u))
    assert isinstance(dvstardu, ZeroBaseForm)
    assert dvstardu.arguments() == vstar.arguments() + (Argument(u.ufl_function_space(), 1),)
    # Check compatibility with int/float
    assert dvstardu == 0

    # -- Matrix -- #
    M = Matrix(V, U)
    dMdu = expand_derivatives(derivative(M, u))
    assert isinstance(dMdu, ZeroBaseForm)
    assert dMdu.arguments() == M.arguments() + (Argument(u.ufl_function_space(), 2),)
    # Check compatibility with int/float
    assert dMdu == 0

    # -- Action -- #
    Ac = Action(w, u)
    dAcdu = derivative(Ac, u)
    assert dAcdu == (action(adjoint(derivative(w, u), derivatives_expanded=True), u, derivatives_expanded=True)
                     + action(w, derivative(u, u), derivatives_expanded=True))

    dAcdu = expand_derivatives(dAcdu)
    # Since dw/du = 0
    assert dAcdu == Action(w, v)

    # -- Form sum -- #
    uhat = Argument(U, 1)
    what = Argument(U, 2)
    Fs = M + inner(u * uhat, v) * dx
    dFsdu = expand_derivatives(derivative(Fs, u))
    # Distribute differentiation over FormSum components
    assert dFsdu == inner(what * uhat, v) * dx


def test_zero_base_form_mult():
    domain_2d = Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), identity_pull_back, H1))
    f_2d = FiniteElement("Lagrange", triangle, 1, (), identity_pull_back, H1)
    V = FunctionSpace(domain_2d, f_2d)
    v = Argument(V, 0)
    Z = ZeroBaseForm((v, v))

    u = Coefficient(V)

    Zu = Z * u
    assert Zu == action(Z, u)
    assert action(Zu, u) == ZeroBaseForm(())
