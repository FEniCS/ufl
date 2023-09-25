__authors__ = "Martin Sandve Alnæs"
__date__ = "2009-03-19 -- 2012-03-20"

# Modified by Anders Logg, 2008
# Modified by Garth N. Wells, 2009

import math

import pytest

from ufl import (Coefficient, FunctionSpace, Identity, Mesh, as_tensor, cos, det, div, dot, dx, exp, grad, i, inner, j,
                 k, l, ln, nabla_div, nabla_grad, outer, sin, triangle)
from ufl.algorithms import compute_form_data, expand_derivatives, expand_indices
from ufl.algorithms.renumbering import renumber_indices
from ufl.finiteelement import FiniteElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1

# TODO: Test expand_indices2 throuroughly for correctness, then efficiency:
# expand_indices, expand_indices2 = expand_indices2, expand_indices


class Fixture:

    def __init__(self):
        cell = triangle
        element = FiniteElement("Lagrange", cell, 1, (), identity_pull_back, H1)
        velement = FiniteElement("Lagrange", cell, 1, (2, ), identity_pull_back, H1)
        telement = FiniteElement("Lagrange", cell, 1, (2, 2), identity_pull_back, H1)
        domain = Mesh(FiniteElement("Lagrange", cell, 1, (2, ), identity_pull_back, H1))
        space = FunctionSpace(domain, element)
        vspace = FunctionSpace(domain, velement)
        tspace = FunctionSpace(domain, telement)
        self.sf = Coefficient(space)
        self.sf2 = Coefficient(space)
        self.vf = Coefficient(vspace)
        self.tf = Coefficient(tspace)

        # Note: the derivatives of these functions make no sense, but
        #       their unique constant values are used for validation.

        def SF(x, derivatives=()):
            # first order derivatives
            if derivatives == (0,):
                return 0.30
            elif derivatives == (1,):
                return 0.31
            # second order derivatives
            elif derivatives == (0, 0):
                return 0
            elif derivatives in ((1, 0), (0, 1)):
                return 0
            elif derivatives == (1, 1):
                return 0
            # function value
            assert derivatives == ()
            return 3

        def SF2(x, derivatives=()):
            # first order derivatives
            if derivatives == (0,):
                return 0.30
            elif derivatives == (1,):
                return 0.31
            # second order derivatives
            elif derivatives == (0, 0):
                return 3.300
            elif derivatives in ((1, 0), (0, 1)):
                return 3.310
            elif derivatives == (1, 1):
                return 3.311
            # function value
            assert derivatives == ()
            return 3

        def VF(x, derivatives=()):
            # first order derivatives
            if derivatives == (0,):
                return (0.50, 0.70)
            elif derivatives == (1,):
                return (0.51, 0.71)
            # second order derivatives
            elif derivatives == (0, 0):
                return (0.20, 0.21)
            elif derivatives in ((1, 0), (0, 1)):
                return (0.30, 0.31)
            elif derivatives == (1, 1):
                return (0.40, 0.41)
            # function value
            assert derivatives == ()
            return (5, 7)

        def TF(x, derivatives=()):
            # first order derivatives
            if derivatives == (0,):
                return ((1.10, 1.30), (1.70, 1.90))
            elif derivatives == (1,):
                return ((1.11, 1.31), (1.71, 1.91))
            # second order derivatives
            elif derivatives == (0, 0):
                return ((10.00, 10.01), (10.10, 10.11))
            elif derivatives in ((1, 0), (0, 1)):
                return ((12.00, 12.01), (12.10, 12.11))
            elif derivatives == (1, 1):
                return ((11.00, 11.01), (11.10, 11.11))
            # function value
            assert derivatives == ()
            return ((11, 13), (17, 19))

        self.x = (1.23, 3.14)
        self.mapping = {self.sf: SF, self.sf2: SF2, self.vf: VF, self.tf: TF}

    def compare(self, f, value):
        debug = 0
        if debug:
            print(('f', f))
        g = expand_derivatives(f)
        if debug:
            print(('g', g))
        gv = g(self.x, self.mapping)
        assert abs(gv - value) < 1e-7

        g = expand_indices(g)
        if debug:
            print(('g', g))
        gv = g(self.x, self.mapping)
        assert abs(gv - value) < 1e-7

        g = renumber_indices(g)
        if debug:
            print(('g', g))
        gv = g(self.x, self.mapping)
        assert abs(gv - value) < 1e-7


@pytest.fixture(scope="module")
def fixt():
    # Workaround for quick pytest transition
    return Fixture()


def test_basic_expand_indices(self, fixt):
    sf = fixt.sf
    vf = fixt.vf
    tf = fixt.tf
    compare = fixt.compare

    # Simple expressions with no indices or derivatives to expand
    compare(sf, 3)
    compare(sf + 1, 4)
    compare(sf - 2.5, 0.5)
    compare(sf/2, 1.5)
    compare(sf/0.5, 6)
    compare(sf**2, 9)
    compare(sf**0.5, 3**0.5)
    compare(sf**3, 27)
    compare(0.5**sf, 0.5**3)
    compare(sf * (sf/6), 1.5)
    compare(sin(sf), math.sin(3))
    compare(cos(sf), math.cos(3))
    compare(exp(sf), math.exp(3))
    compare(ln(sf), math.log(3))

    # Simple indexing
    compare(vf[0], 5)
    compare(vf[0] + 1, 6)
    compare(vf[0] - 2.5, 2.5)
    compare(vf[0]/2, 2.5)
    compare(vf[0]/0.5, 10)
    compare(vf[0]**2, 25)
    compare(vf[0]**0.5, 5**0.5)
    compare(vf[0]**3, 125)
    compare(0.5**vf[0], 0.5**5)
    compare(vf[0] * (vf[0]/6), 5*(5./6))
    compare(sin(vf[0]), math.sin(5))
    compare(cos(vf[0]), math.cos(5))
    compare(exp(vf[0]), math.exp(5))
    compare(ln(vf[0]), math.log(5))

    # Double indexing
    compare(tf[1, 1], 19)
    compare(tf[1, 1] + 1, 20)
    compare(tf[1, 1] - 2.5, 16.5)
    compare(tf[1, 1]/2, 9.5)
    compare(tf[1, 1]/0.5, 38)
    compare(tf[1, 1]**2, 19**2)
    compare(tf[1, 1]**0.5, 19**0.5)
    compare(tf[1, 1]**3, 19**3)
    compare(0.5**tf[1, 1], 0.5**19)
    compare(tf[1, 1] * (tf[1, 1]/6), 19*(19./6))
    compare(sin(tf[1, 1]), math.sin(19))
    compare(cos(tf[1, 1]), math.cos(19))
    compare(exp(tf[1, 1]), math.exp(19))
    compare(ln(tf[1, 1]), math.log(19))


def test_expand_indices_index_sum(self, fixt):
    vf = fixt.vf
    tf = fixt.tf
    compare = fixt.compare

    # Basic index sums
    compare(vf[i]*vf[i], 5*5+7*7)
    compare(vf[j]*tf[i, j]*vf[i], 5*5*11 + 5*7*13 + 5*7*17 + 7*7*19)
    compare(vf[j]*tf.T[j, i]*vf[i], 5*5*11 + 5*7*13 + 5*7*17 + 7*7*19)
    compare(tf[i, i], 11 + 19)
    compare(tf[i, j]*(tf[j, i]+outer(vf, vf)[i, j]), (5*5+11)*11 + (7*5+17)*13 + (7*5+13)*17 + (7*7+19)*19)
    compare(as_tensor(as_tensor(tf[i, j], (i, j))[k, l], (l, k))[i, i], 11 + 19)


def test_expand_indices_derivatives(self, fixt):
    sf = fixt.sf
    vf = fixt.vf
    compare = fixt.compare

    # Basic derivatives
    compare(sf.dx(0), 0.3)
    compare(sf.dx(1), 0.31)
    compare(sf.dx(i)*vf[i], 0.30*5 + 0.31*7)
    compare(vf[j].dx(i)*vf[i].dx(j), 0.50*0.50 + 0.51*0.70 + 0.70*0.51 + 0.71*0.71)


def test_expand_indices_hyperelasticity(self, fixt):
    vf = fixt.vf
    compare = fixt.compare

    # Deformation gradient
    ident = Identity(2)
    u = vf
    F = ident + grad(u)
    # F = (1 + vf[0].dx(0), vf[0].dx(1), vf[1].dx(0), 1 + vf[1].dx(1))
    # F = (1 + 0.50,        0.51,        0.70,        1 + 0.71)
    F00 = 1 + 0.50
    F01 = 0.51
    F10 = 0.70
    F11 = 1 + 0.71
    compare(F[0, 0], F00)
    compare(F[0, 1], F01)
    compare(F[1, 0], F10)
    compare(F[1, 1], F11)

    J = det(F)
    compare(J, (1 + 0.50)*(1 + 0.71) - 0.70*0.51)

    # Strain tensors
    C = F.T*F
    # Cij = sum_k Fki Fkj
    C00 = F00*F00 + F10*F10
    C01 = F00*F01 + F10*F11
    C10 = F01*F00 + F11*F10
    C11 = F01*F01 + F11*F11
    compare(C[0, 0], C00)
    compare(C[0, 1], C01)
    compare(C[1, 0], C10)
    compare(C[1, 1], C11)

    E = (C-ident)/2
    E00 = (C00-1)/2
    E01 = (C01)/2
    E10 = (C10)/2
    E11 = (C11-1)/2
    compare(E[0, 0], E00)
    compare(E[0, 1], E01)
    compare(E[1, 0], E10)
    compare(E[1, 1], E11)

    # Strain energy
    Q = inner(E, E)
    Qvalue = E00**2 + E01**2 + E10**2 + E11**2
    compare(Q, Qvalue)

    K = 0.5
    psi = (K/2)*exp(Q)
    compare(psi, 0.25*math.exp(Qvalue))


def test_expand_indices_div_grad(self, fixt):
    sf = fixt.sf
    sf2 = fixt.sf2
    vf = fixt.vf
    tf = fixt.tf
    compare = fixt.compare

    a = div(grad(sf))
    compare(a, 0)

    a = div(grad(sf2))
    compare(a, 3.300 + 3.311)

    if 0:
        Dvf = grad(vf)
        Lvf = div(Dvf)
        Lvf2 = dot(Lvf, Lvf)
        pp = compute_form_data(Lvf2*dx).preprocessed_form.integrals()[0].integrand()
        print(('vf', vf.ufl_shape, str(vf)))
        print(('Dvf', Dvf.ufl_shape, str(Dvf)))
        print(('Lvf', Lvf.ufl_shape, str(Lvf)))
        print(('Lvf2', Lvf2.ufl_shape, str(Lvf2)))
        print(('pp', pp.ufl_shape, str(pp)))

    a = div(grad(vf))
    compare(dot(a, a), (0.20+0.40)**2 + (0.21+0.41)**2)

    a = div(grad(tf))
    compare(inner(a, a), (10.00+11.00)**2 + (10.01+11.01)**2 + (10.10+11.10)**2 + (10.11+11.11)**2)


def test_expand_indices_nabla_div_grad(self, fixt):
    sf = fixt.sf
    sf2 = fixt.sf2
    vf = fixt.vf
    tf = fixt.tf
    compare = fixt.compare

    a = nabla_div(nabla_grad(sf))
    compare(a, 0)

    a = nabla_div(nabla_grad(sf2))
    compare(a, 3.300 + 3.311)

    a = nabla_div(nabla_grad(vf))
    compare(dot(a, a), (0.20+0.40)**2 + (0.21+0.41)**2)

    a = nabla_div(nabla_grad(tf))
    compare(inner(a, a), (10.00+11.00)**2 + (10.01+11.01)**2 + (10.10+11.10)**2 + (10.11+11.11)**2)
