#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

from __future__ import print_function

from pytest import raises
from ufl import *
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks, apply_single_function_pullbacks
from ufl.algorithms.renumbering import renumber_indices
from ufl.classes import Jacobian, JacobianInverse, JacobianDeterminant, ReferenceValue, CellOrientation

def check_single_function_pullback(g, mappings):
    expected = mappings[g]
    actual = apply_single_function_pullbacks(g)
    rexp = renumber_indices(expected)
    ract = renumber_indices(actual)
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

    U0 = FiniteElement("DG", cell, 0)
    U = FiniteElement("CG", cell, 1)
    V = VectorElement("CG", cell, 1)
    Vd = FiniteElement("RT", cell, 1)
    Vc = FiniteElement("N1curl", cell, 1)
    T = TensorElement("CG", cell, 1)
    S = TensorElement("CG", cell, 1, symmetry=True)

    Um = U*U
    Vm = U*V
    Vdm = V*Vd
    Vcm = Vd*Vc
    Tm = Vc*T
    Sm = T*S

    Vd0 = Vd*U0 # case from failing ffc demo

    W = S*T*Vc*Vd*V*U

    u = Coefficient(U)
    v = Coefficient(V)
    vd = Coefficient(Vd)
    vc = Coefficient(Vc)
    t = Coefficient(T)
    s = Coefficient(S)

    um = Coefficient(Um)
    vm = Coefficient(Vm)
    vdm = Coefficient(Vdm)
    vcm = Coefficient(Vcm)
    tm = Coefficient(Tm)
    sm = Coefficient(Sm)

    vd0m = Coefficient(Vd0) # case from failing ffc demo

    w = Coefficient(W)

    ru = ReferenceValue(u)
    rv = ReferenceValue(v)
    rvd = ReferenceValue(vd)
    rvc = ReferenceValue(vc)
    rt = ReferenceValue(t)
    rs = ReferenceValue(s)

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
    #o = CellOrientation(domain)
    i, j, k, l = indices(4)

    # Contravariant H(div) Piola mapping:
    M_hdiv = ((1.0/detJ) * J) # Not applying cell orientation here
    # Covariant H(curl) Piola mapping: Jinv.T

    mappings = {
        # Simple elements should get a simple representation
        u: ru,
        v: rv,
        vd: as_vector(M_hdiv[i,j]*rvd[j], i),
        vc: as_vector(Jinv[j,i]*rvc[j], i),
        t: rt,
        s: as_tensor([[rs[0], rs[1], rs[2]],
                      [rs[1], rs[3], rs[4]],
                      [rs[2], rs[4], rs[5]]]),
        # Mixed elements become a bit more complicated
        um: rum,
        vm: rvm,
        vdm: as_vector([
            # V
            rvdm[0],
            rvdm[1],
            rvdm[2],
            # Vd
            M_hdiv[0,j]*as_vector([rvdm[3], rvdm[4]])[j],
            M_hdiv[1,j]*as_vector([rvdm[3], rvdm[4]])[j],
            M_hdiv[2,j]*as_vector([rvdm[3], rvdm[4]])[j],
            ]),
        vcm: as_vector([
            # Vd
            M_hdiv[0,j]*as_vector([rvcm[0], rvcm[1]])[j],
            M_hdiv[1,j]*as_vector([rvcm[0], rvcm[1]])[j],
            M_hdiv[2,j]*as_vector([rvcm[0], rvcm[1]])[j],
            # Vc
            Jinv[i,0]*as_vector([rvcm[2], rvcm[3]])[i],
            Jinv[i,1]*as_vector([rvcm[2], rvcm[3]])[i],
            Jinv[i,2]*as_vector([rvcm[2], rvcm[3]])[i],
            ]),
        tm: as_vector([
            # Vc
            Jinv[i,0]*as_vector([rtm[0], rtm[1]])[i],
            Jinv[i,1]*as_vector([rtm[0], rtm[1]])[i],
            Jinv[i,2]*as_vector([rtm[0], rtm[1]])[i],
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
            rsm[ 9], rsm[10], rsm[11],
            rsm[10], rsm[12], rsm[13],
            rsm[11], rsm[13], rsm[14],
            ]),
        # Case from failing ffc demo:
        vd0m: as_vector([
            M_hdiv[0,j]*as_vector([rvd0m[0],rvd0m[1]])[j],
            M_hdiv[1,j]*as_vector([rvd0m[0],rvd0m[1]])[j],
            M_hdiv[2,j]*as_vector([rvd0m[0],rvd0m[1]])[j],
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
            Jinv[i,0]*as_vector([rw[15], rw[16]])[i],
            Jinv[i,1]*as_vector([rw[15], rw[16]])[i],
            Jinv[i,2]*as_vector([rw[15], rw[16]])[i],
            # Vd
            M_hdiv[0,j]*as_vector([rw[17], rw[18]])[j],
            M_hdiv[1,j]*as_vector([rw[17], rw[18]])[j],
            M_hdiv[2,j]*as_vector([rw[17], rw[18]])[j],
            # V
            rw[19],
            rw[20],
            rw[21],
            # U
            rw[22],
            ]),
        }

    # Check functions of various elements outside a mixed context
    check_single_function_pullback(u, mappings)
    check_single_function_pullback(v, mappings)
    check_single_function_pullback(vd, mappings)
    check_single_function_pullback(vc, mappings)
    check_single_function_pullback(t, mappings)
    check_single_function_pullback(s, mappings)

    # Check functions of various elements inside a mixed context
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

    U = FiniteElement("CG", cell, 1)
    V = VectorElement("CG", cell, 1)
    Vd = FiniteElement("RT", cell, 1)
    Vc = FiniteElement("N1curl", cell, 1)
    T = TensorElement("CG", cell, 1)
    S = TensorElement("CG", cell, 1, symmetry=True)

    Um = U*U
    Vm = U*V
    Vdm = V*Vd
    Vcm = Vd*Vc
    Tm = Vc*T
    Sm = T*S

    W = S*T*Vc*Vd*V*U

    u = Coefficient(U)
    v = Coefficient(V)
    vd = Coefficient(Vd)
    vc = Coefficient(Vc)
    t = Coefficient(T)
    s = Coefficient(S)

    um = Coefficient(Um)
    vm = Coefficient(Vm)
    vdm = Coefficient(Vdm)
    vcm = Coefficient(Vcm)
    tm = Coefficient(Tm)
    sm = Coefficient(Sm)

    w = Coefficient(W)

    ru = ReferenceValue(u)
    rv = ReferenceValue(v)
    rvd = ReferenceValue(vd)
    rvc = ReferenceValue(vc)
    rt = ReferenceValue(t)
    rs = ReferenceValue(s)

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
        u: ru,
        v: rv,
        vd: as_vector(M_hdiv[i,j]*rvd[j], i),
        vc: as_vector(Jinv[j,i]*rvc[j], i),
        t: rt,
        s: as_tensor([[rs[0], rs[1]], [rs[1], rs[2]]]),
        # Mixed elements become a bit more complicated
        um: rum,
        vm: rvm,
        vdm: as_vector([
            # V
            rvdm[0],
            rvdm[1],
            # Vd
            M_hdiv[0,j]*as_vector([rvdm[2], rvdm[3]])[j],
            M_hdiv[1,j]*as_vector([rvdm[2], rvdm[3]])[j],
            ]),
        vcm: as_vector([
            # Vd
            M_hdiv[0,j]*as_vector([rvcm[0], rvcm[1]])[j],
            M_hdiv[1,j]*as_vector([rvcm[0], rvcm[1]])[j],
            # Vc
            Jinv[i,0]*as_vector([rvcm[2], rvcm[3]])[i],
            Jinv[i,1]*as_vector([rvcm[2], rvcm[3]])[i],
            ]),
        tm: as_vector([
            # Vc
            Jinv[i,0]*as_vector([rtm[0], rtm[1]])[i],
            Jinv[i,1]*as_vector([rtm[0], rtm[1]])[i],
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
            Jinv[i,0]*as_vector([rw[7], rw[8]])[i],
            Jinv[i,1]*as_vector([rw[7], rw[8]])[i],
            # Vd
            M_hdiv[0,j]*as_vector([rw[9], rw[10]])[j],
            M_hdiv[1,j]*as_vector([rw[9], rw[10]])[j],
            # V
            rw[11],
            rw[12],
            # U
            rw[13],
            ]),
        }

    # Check functions of various elements outside a mixed context
    check_single_function_pullback(u, mappings)
    check_single_function_pullback(v, mappings)
    check_single_function_pullback(vd, mappings)
    check_single_function_pullback(vc, mappings)
    check_single_function_pullback(t, mappings)
    check_single_function_pullback(s, mappings)

    # Check functions of various elements inside a mixed context
    check_single_function_pullback(um, mappings)
    check_single_function_pullback(vm, mappings)
    check_single_function_pullback(vdm, mappings)
    check_single_function_pullback(vcm, mappings)
    check_single_function_pullback(tm, mappings)
    check_single_function_pullback(sm, mappings)

    # Check the ridiculous mixed element W combining it all
    check_single_function_pullback(w, mappings)
