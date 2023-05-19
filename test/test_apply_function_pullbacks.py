#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

import numpy

from ufl import Cell, Coefficient, as_domain, as_tensor, as_vector, indices, triangle
from ufl.algorithms.apply_function_pullbacks import apply_single_function_pullbacks
from ufl.algorithms.renumbering import renumber_indices
from ufl.classes import CellOrientation, Jacobian, JacobianDeterminant, JacobianInverse, ReferenceValue
from ufl.finiteelement import FiniteElement, MixedElement
from ufl.sobolevspace import H1, L2, HCurl, HDiv, HDivDiv, HEin


def check_single_function_pullback(g, mappings):
    expected = mappings[g]
    actual = apply_single_function_pullbacks(ReferenceValue(g), g.ufl_element())
    assert expected.ufl_shape == actual.ufl_shape
    for idx in numpy.ndindex(actual.ufl_shape):
        rexp = renumber_indices(expected[idx])
        ract = renumber_indices(actual[idx])
        if not rexp == ract:
            print()
            print("In check_single_function_pullback:")
            print("input:")
            print(repr(g))
            print("expected:")
            print(str(rexp))
            print("actual:")
            print(str(ract))
            print("signatures:")
            print((expected**2*dx).signature())
            print((actual**2*dx).signature())
            print()
        assert ract == rexp


def test_apply_single_function_pullbacks_triangle3d():
    triangle3d = Cell("triangle", geometric_dimension=3)
    cell = triangle3d
    domain = as_domain(cell)

    UL2 = FiniteElement("Discontinuous Lagrange", cell, 1, (), (), "L2 Piola", L2)
    U0 = FiniteElement("Discontinuous Lagrange", cell, 0, (), (), "identity", L2)
    U = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    V = FiniteElement("Lagrange", cell, 1, (3, ), (3, ), "identity", H1)
    Vd = FiniteElement("Raviart-Thomas", cell, 1, (3, ), (2, ), "contravariant Piola", HDiv)
    Vc = FiniteElement("N1curl", cell, 1, (3, ), (2, ), "covariant Piola", HCurl)
    T = FiniteElement("Lagrange", cell, 1, (3, 3), (3, 3), "identity", H1)
    S = FiniteElement("Lagrange", cell, 1, (3, 3), (6, ), "symmetries", H1, component_map={
        (0, 0): 0, (1, 0): 1, (2, 0): 2, (0, 1): 1, (1, 1): 3, (2, 1): 4, (0, 2): 2, (1, 2): 4, (2, 2): 5
    }, sub_elements=[
        FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
        for _ in range(6)])
    COV2T = FiniteElement("Regge", cell, 0, (3, 3), (2, 2), "double covariant Piola", HEin)   # (0, 2)-symmetric tensors
    CONTRA2T = FiniteElement("HHJ", cell, 0, (3, 3), (2, 2), "double contravariant Piola", HDivDiv)  # (2, 0)-symmetric tensors

    Uml2 = MixedElement([UL2, UL2])
    Um = MixedElement([U, U])
    Vm = MixedElement([U, V])
    Vdm = MixedElement([V, Vd])
    Vcm = MixedElement([Vd, Vc])
    Tm = MixedElement([Vc, T])
    Sm = MixedElement([T, S])

    Vd0 = MixedElement([Vd, U0])  # case from failing ffc demo

    W = MixedElement([S, T, Vc, Vd, V, U])

    ul2 = Coefficient(UL2)
    u = Coefficient(U)
    v = Coefficient(V)
    vd = Coefficient(Vd)
    vc = Coefficient(Vc)
    t = Coefficient(T)
    s = Coefficient(S)
    cov2t = Coefficient(COV2T)
    contra2t = Coefficient(CONTRA2T)

    uml2 = Coefficient(Uml2)
    um = Coefficient(Um)
    vm = Coefficient(Vm)
    vdm = Coefficient(Vdm)
    vcm = Coefficient(Vcm)
    tm = Coefficient(Tm)
    sm = Coefficient(Sm)

    vd0m = Coefficient(Vd0)  # case from failing ffc demo

    w = Coefficient(W)

    rul2 = ReferenceValue(ul2)
    ru = ReferenceValue(u)
    rv = ReferenceValue(v)
    rvd = ReferenceValue(vd)
    rvc = ReferenceValue(vc)
    rt = ReferenceValue(t)
    rs = ReferenceValue(s)
    rcov2t = ReferenceValue(cov2t)
    rcontra2t = ReferenceValue(contra2t)

    ruml2 = ReferenceValue(uml2)
    rum = ReferenceValue(um)
    rvm = ReferenceValue(vm)
    rvdm = ReferenceValue(vdm)
    rvcm = ReferenceValue(vcm)
    rtm = ReferenceValue(tm)
    rsm = ReferenceValue(sm)

    rvd0m = ReferenceValue(vd0m)

    rw = ReferenceValue(w)
    assert len(w) == 9 + 9 + 3 + 3 + 3 + 1
    assert len(rw) == 6 + 9 + 2 + 2 + 3 + 1
    assert len(w) == 28
    assert len(rw) == 23

    assert len(vd0m) == 4
    assert len(rvd0m) == 3

    # Geometric quantities we need:
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)
    # o = CellOrientation(domain)
    i, j, k, l = indices(4)

    # Contravariant H(div) Piola mapping:
    M_hdiv = ((1.0/detJ) * J)  # Not applying cell orientation here
    # Covariant H(curl) Piola mapping: Jinv.T

    mappings = {
        # Simple elements should get a simple representation
        ul2: rul2 / detJ,
        u: ru,
        v: rv,
        vd: as_vector(M_hdiv[i, j]*rvd[j], i),
        vc: as_vector(Jinv[j, i]*rvc[j], i),
        t: rt,
        s: as_tensor([[rs[0], rs[1], rs[2]],
                      [rs[1], rs[3], rs[4]],
                      [rs[2], rs[4], rs[5]]]),
        cov2t: as_tensor(Jinv[k, i] * rcov2t[k, l] * Jinv[l, j], (i, j)),
        contra2t: as_tensor((1.0 / detJ)**2
                            * J[i, k] * rcontra2t[k, l] * J[j, l], (i, j)),
        # Mixed elements become a bit more complicated
        uml2: as_vector([ruml2[0] / detJ, ruml2[1] / detJ]),
        um: rum,
        vm: rvm,
        vdm: as_vector([
            # V
            rvdm[0],
            rvdm[1],
            rvdm[2],
            # Vd
            *(as_tensor(M_hdiv[i, j]*as_vector([rvdm[3], rvdm[4]])[j], (i,))[n]
              for n in range(3))
    ]),
        vcm: as_vector([
            # Vd
            *(as_tensor(M_hdiv[i, j]*as_vector([rvcm[0], rvcm[1]])[j], (i,))[n]
              for n in range(3)),
            # Vc
            *(as_tensor(Jinv[i, j]*as_vector([rvcm[2], rvcm[3]])[i], (j,))[n]
              for n in range(3))
    ]),
        tm: as_vector([
            # Vc
            *(as_tensor(Jinv[i, j]*as_vector([rtm[0], rtm[1]])[i], (j,))[n]
              for n in range(3)),
            # T
            rtm[2], rtm[3], rtm[4],
            rtm[5], rtm[6], rtm[7],
            rtm[8], rtm[9], rtm[10],
    ]),
        sm: as_vector([
            # T
            rsm[0], rsm[1], rsm[2],
            rsm[3], rsm[4], rsm[5],
            rsm[6], rsm[7], rsm[8],
            # S
            rsm[9], rsm[10], rsm[11],
            rsm[10], rsm[12], rsm[13],
            rsm[11], rsm[13], rsm[14],
    ]),
        # Case from failing ffc demo:
        vd0m: as_vector([
            M_hdiv[0, j]*as_vector([rvd0m[0], rvd0m[1]])[j],
            M_hdiv[1, j]*as_vector([rvd0m[0], rvd0m[1]])[j],
            M_hdiv[2, j]*as_vector([rvd0m[0], rvd0m[1]])[j],
            rvd0m[2]
    ]),
        # This combines it all:
        w: as_vector([
            # S
            rw[0], rw[1], rw[2],
            rw[1], rw[3], rw[4],
            rw[2], rw[4], rw[5],
            # T
            rw[6], rw[7], rw[8],
            rw[9], rw[10], rw[11],
            rw[12], rw[13], rw[14],
            # Vc
            *(as_tensor(Jinv[i, j]*as_vector([rw[15], rw[16]])[i], (j,))[n]
              for n in range(3)),
            # Vd
            *(as_tensor(M_hdiv[i, j]*as_vector([rw[17], rw[18]])[j], (i,))[n]
              for n in range(3)),
            # V
            rw[19],
            rw[20],
            rw[21],
            # U
            rw[22],
    ]),
    }

    # Check functions of various elements outside a mixed context
    check_single_function_pullback(ul2, mappings)
    check_single_function_pullback(u, mappings)
    check_single_function_pullback(v, mappings)
    check_single_function_pullback(vd, mappings)
    check_single_function_pullback(vc, mappings)
    check_single_function_pullback(t, mappings)
    check_single_function_pullback(s, mappings)
    check_single_function_pullback(cov2t, mappings)
    check_single_function_pullback(contra2t, mappings)

    # Check functions of various elements inside a mixed context
    check_single_function_pullback(uml2, mappings)
    check_single_function_pullback(um, mappings)
    check_single_function_pullback(vm, mappings)
    check_single_function_pullback(vdm, mappings)
    check_single_function_pullback(vcm, mappings)
    check_single_function_pullback(tm, mappings)
    check_single_function_pullback(sm, mappings)

    # Check the ridiculous mixed element W combining it all
    check_single_function_pullback(w, mappings)


def test_apply_single_function_pullbacks_triangle():
    cell = triangle
    domain = as_domain(cell)

    Ul2 = FiniteElement("Discontinuous Lagrange", cell, 1, (), (), "L2 Piola", L2)
    U = FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
    V = FiniteElement("Lagrange", cell, 1, (2, ), (2, ), "identity", H1)
    Vd = FiniteElement("Raviart-Thomas", cell, 1, (2, ), (2, ), "contravariant Piola", HDiv)
    Vc = FiniteElement("N1curl", cell, 1, (2, ), (2, ), "covariant Piola", HCurl)
    T = FiniteElement("Lagrange", cell, 1, (2, 2), (2, 2), "identity", H1)
    S = FiniteElement("Lagrange", cell, 1, (2, 2), (3, ), "symmetries", H1, component_map={
        (0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2
    }, sub_elements=[
        FiniteElement("Lagrange", cell, 1, (), (), "identity", H1)
        for i in range(3)])

    Uml2 = MixedElement([Ul2, Ul2])
    Um = MixedElement([U, U])
    Vm = MixedElement([U, V])
    Vdm = MixedElement([V, Vd])
    Vcm = MixedElement([Vd, Vc])
    Tm = MixedElement([Vc, T])
    Sm = MixedElement([T, S])

    W = MixedElement([S, T, Vc, Vd, V, U])

    ul2 = Coefficient(Ul2)
    u = Coefficient(U)
    v = Coefficient(V)
    vd = Coefficient(Vd)
    vc = Coefficient(Vc)
    t = Coefficient(T)
    s = Coefficient(S)

    uml2 = Coefficient(Uml2)
    um = Coefficient(Um)
    vm = Coefficient(Vm)
    vdm = Coefficient(Vdm)
    vcm = Coefficient(Vcm)
    tm = Coefficient(Tm)
    sm = Coefficient(Sm)

    w = Coefficient(W)

    rul2 = ReferenceValue(ul2)
    ru = ReferenceValue(u)
    rv = ReferenceValue(v)
    rvd = ReferenceValue(vd)
    rvc = ReferenceValue(vc)
    rt = ReferenceValue(t)
    rs = ReferenceValue(s)

    ruml2 = ReferenceValue(uml2)
    rum = ReferenceValue(um)
    rvm = ReferenceValue(vm)
    rvdm = ReferenceValue(vdm)
    rvcm = ReferenceValue(vcm)
    rtm = ReferenceValue(tm)
    rsm = ReferenceValue(sm)

    rw = ReferenceValue(w)

    assert len(w) == 4 + 4 + 2 + 2 + 2 + 1
    assert len(rw) == 3 + 4 + 2 + 2 + 2 + 1
    assert len(w) == 15
    assert len(rw) == 14

    # Geometric quantities we need:
    J = Jacobian(domain)
    detJ = JacobianDeterminant(domain)
    Jinv = JacobianInverse(domain)
    i, j, k, l = indices(4)

    # Contravariant H(div) Piola mapping:
    M_hdiv = (1.0/detJ) * J
    # Covariant H(curl) Piola mapping: Jinv.T

    mappings = {
        # Simple elements should get a simple representation
        ul2: rul2 / detJ,
        u: ru,
        v: rv,
        vd: as_vector(M_hdiv[i, j]*rvd[j], i),
        vc: as_vector(Jinv[j, i]*rvc[j], i),
        t: rt,
        s: as_tensor([[rs[0], rs[1]], [rs[1], rs[2]]]),
        # Mixed elements become a bit more complicated
        uml2: as_vector([ruml2[0] / detJ, ruml2[1] / detJ]),
        um: rum,
        vm: rvm,
        vdm: as_vector([
            # V
            rvdm[0],
            rvdm[1],
            # Vd
            *(as_tensor(M_hdiv[i, j]*as_vector([rvdm[2], rvdm[3]])[j], (i,))[n]
              for n in range(2)),
    ]),
        vcm: as_vector([
            # Vd
            *(as_tensor(M_hdiv[i, j]*as_vector([rvcm[0], rvcm[1]])[j], (i,))[n]
              for n in range(2)),
            # Vc
            *(as_tensor(Jinv[i, j]*as_vector([rvcm[2], rvcm[3]])[i], (j,))[n]
              for n in range(2)),
    ]),
        tm: as_vector([
            # Vc
            *(as_tensor(Jinv[i, j]*as_vector([rtm[0], rtm[1]])[i], (j,))[n]
              for n in range(2)),
            # T
            rtm[2], rtm[3],
            rtm[4], rtm[5],
    ]),
        sm: as_vector([
            # T
            rsm[0], rsm[1],
            rsm[2], rsm[3],
            # S
            rsm[4], rsm[5],
            rsm[5], rsm[6],
    ]),
        # This combines it all:
        w: as_vector([
            # S
            rw[0], rw[1],
            rw[1], rw[2],
            # T
            rw[3], rw[4],
            rw[5], rw[6],
            # Vc
            *(as_tensor(Jinv[i, j]*as_vector([rw[7], rw[8]])[i], (j,))[n]
              for n in range(2)),
            # Vd
            *(as_tensor(M_hdiv[i, j]*as_vector([rw[9], rw[10]])[j], (i,))[n]
              for n in range(2)),
            # V
            rw[11],
            rw[12],
            # U
            rw[13],
    ]),
    }

    # Check functions of various elements outside a mixed context
    check_single_function_pullback(ul2, mappings)
    check_single_function_pullback(u, mappings)
    check_single_function_pullback(v, mappings)
    check_single_function_pullback(vd, mappings)
    check_single_function_pullback(vc, mappings)
    check_single_function_pullback(t, mappings)
    check_single_function_pullback(s, mappings)

    # Check functions of various elements inside a mixed context
    check_single_function_pullback(uml2, mappings)
    check_single_function_pullback(um, mappings)
    check_single_function_pullback(vm, mappings)
    check_single_function_pullback(vdm, mappings)
    check_single_function_pullback(vcm, mappings)
    check_single_function_pullback(tm, mappings)
    check_single_function_pullback(sm, mappings)

    # Check the ridiculous mixed element W combining it all
    check_single_function_pullback(w, mappings)
