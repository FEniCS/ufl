#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

__authors__ = "David Ham"
__date__ = "2014-03-04"

import pytest
from ufl import (EnrichedElement, TensorProductElement,
                 FiniteElement, triangle, interval,
                 quadrilateral, HDiv, HCurl)
from ufl.sobolevspace import SobolevSpace, DirectionalSobolevSpace
from ufl import H2, H1, HDiv, HCurl, L2


# Construct directional Sobolev spaces, with varying smoothness in
# spatial coordinates
H0dx0dy = DirectionalSobolevSpace((0, 0))
H1dx1dy = DirectionalSobolevSpace((1, 1))
H2dx2dy = DirectionalSobolevSpace((2, 2))
H1dx = DirectionalSobolevSpace((1, 0))
H1dy = DirectionalSobolevSpace((0, 1))
H000 = DirectionalSobolevSpace((0, 0, 0))
H1dz = DirectionalSobolevSpace((0, 0, 1))
H1dh = DirectionalSobolevSpace((1, 1, 0))
H2dhH1dz = DirectionalSobolevSpace((2, 2, 1))

# TODO: Add construction of all elements with periodic table notation here.


def test_inclusion():
    assert H2 < H1       # Inclusion
    assert not H2 > H1   # Not included
    assert HDiv <= HDiv  # Reflexivity
    assert H2 < L2       # Transitivity


def test_directional_space_relations():
    assert H0dx0dy == L2
    assert H1dx1dy == H1
    assert H2dx2dy == H2
    assert H1dx1dy <= HDiv
    assert H1dx1dy <= HCurl
    assert H2dx2dy <= H1dx1dy
    assert H2dhH1dz < H1
    assert H1dz > H2dhH1dz
    assert H1dh < L2
    assert H1dz < L2
    assert L2 > H1dx
    assert L2 > H1dy
    assert not H1dh <= HDiv
    assert not H1dh <= HCurl


def test_repr():
    assert eval(repr(H2)) == H2


def xtest_contains_mixed():
    pass  # FIXME: How to handle this?


def test_contains_l2():
    l2_elements = [
        FiniteElement("Real", triangle, 0),
        FiniteElement("DG", triangle, 0),
        FiniteElement("DG", triangle, 1),
        FiniteElement("DG", triangle, 2),
        FiniteElement("CR", triangle, 1),
        # Tensor product elements:
        TensorProductElement(FiniteElement("DG", interval, 1),
                             FiniteElement("DG", interval, 1)),
        TensorProductElement(FiniteElement("DG", interval, 1),
                             FiniteElement("CG", interval, 2)),
        # Enriched element:
        EnrichedElement(FiniteElement("DG", triangle, 1),
                        FiniteElement("B", triangle, 3))
    ]
    for l2_element in l2_elements:
        assert l2_element in L2
        assert l2_element in H0dx0dy
        assert l2_element not in H1
        assert l2_element not in H1dx1dy
        assert l2_element not in HCurl
        assert l2_element not in HDiv
        assert l2_element not in H2
        assert l2_element not in H2dx2dy


def test_contains_h1():
    h1_elements = [
        # Standard Lagrange elements:
        FiniteElement("CG", triangle, 1),
        FiniteElement("CG", triangle, 2),
        # Some special elements:
        FiniteElement("AW", triangle),
        FiniteElement("HER", triangle),
        FiniteElement("MTW", triangle),
        # Tensor product elements:
        TensorProductElement(FiniteElement("CG", interval, 1),
                             FiniteElement("CG", interval, 1)),
        TensorProductElement(FiniteElement("CG", interval, 2),
                             FiniteElement("CG", interval, 2)),
        # Enriched elements:
        EnrichedElement(FiniteElement("CG", triangle, 2),
                        FiniteElement("B", triangle, 3))
    ]
    for h1_element in h1_elements:
        assert h1_element in H1
        assert h1_element in H1dx1dy
        assert h1_element in HDiv
        assert h1_element in HCurl
        assert h1_element in L2
        assert h1_element in H0dx0dy
        assert h1_element not in H2
        assert h1_element not in H2dx2dy


def test_contains_h2():
    h2_elements = [
        FiniteElement("ARG", triangle, 1),
        FiniteElement("MOR", triangle),
    ]
    for h2_element in h2_elements:
        assert h2_element in H2
        assert h2_element in H2dx2dy
        assert h2_element in H1
        assert h2_element in H1dx1dy
        assert h2_element in HDiv
        assert h2_element in HCurl
        assert h2_element in L2
        assert h2_element in H0dx0dy


def test_contains_hdiv():
    hdiv_elements = [
        FiniteElement("RT", triangle, 1),
        FiniteElement("BDM", triangle, 1),
        FiniteElement("BDFM", triangle, 2),
        # HDiv elements:
        HDiv(TensorProductElement(FiniteElement("DG", triangle, 1),
                                  FiniteElement("CG", interval, 2))),
        HDiv(TensorProductElement(FiniteElement("RT", triangle, 1),
                                  FiniteElement("DG", interval, 1))),
        HDiv(TensorProductElement(FiniteElement("N1curl", triangle, 1),
                                  FiniteElement("DG", interval, 1)))
    ]
    for hdiv_element in hdiv_elements:
        assert hdiv_element in HDiv
        assert hdiv_element in L2
        assert hdiv_element in H0dx0dy
        assert hdiv_element not in H1
        assert hdiv_element not in H1dx1dy
        assert hdiv_element not in HCurl
        assert hdiv_element not in H2
        assert hdiv_element not in H2dx2dy


def test_contains_hcurl():
    hcurl_elements = [
        FiniteElement("N1curl", triangle, 1),
        FiniteElement("N2curl", triangle, 1),
        # HCurl elements:
        HCurl(TensorProductElement(FiniteElement("CG", triangle, 1),
                                   FiniteElement("DG", interval, 1))),
        HCurl(TensorProductElement(FiniteElement("N1curl", triangle, 1),
                                   FiniteElement("CG", interval, 1))),
        HCurl(TensorProductElement(FiniteElement("RT", triangle, 1),
                                   FiniteElement("CG", interval, 1)))
    ]
    for hcurl_element in hcurl_elements:
        assert hcurl_element in HCurl
        assert hcurl_element in L2
        assert hcurl_element in H0dx0dy
        assert hcurl_element not in H1
        assert hcurl_element not in H1dx1dy
        assert hcurl_element not in HDiv
        assert hcurl_element not in H2
        assert hcurl_element not in H2dx2dy


def test_enriched_elements_hdiv():
    A = FiniteElement("CG", interval, 1)
    B = FiniteElement("DG", interval, 0)
    AxB = TensorProductElement(A, B)
    BxA = TensorProductElement(B, A)
    C = FiniteElement("RTCF", quadrilateral, 1)
    D = FiniteElement("DQ", quadrilateral, 0)
    Q1 = TensorProductElement(C, B)
    Q2 = TensorProductElement(D, A)
    hdiv_elements = [
        EnrichedElement(HDiv(AxB), HDiv(BxA)),
        EnrichedElement(HDiv(Q1), HDiv(Q2))
    ]
    for hdiv_element in hdiv_elements:
        assert hdiv_element in HDiv
        assert hdiv_element in L2
        assert hdiv_element in H0dx0dy
        assert hdiv_element not in H1
        assert hdiv_element not in H1dx1dy
        assert hdiv_element not in HCurl
        assert hdiv_element not in H2
        assert hdiv_element not in H2dx2dy


def test_enriched_elements_hcurl():
    A = FiniteElement("CG", interval, 1)
    B = FiniteElement("DG", interval, 0)
    AxB = TensorProductElement(A, B)
    BxA = TensorProductElement(B, A)
    C = FiniteElement("RTCE", quadrilateral, 1)
    D = FiniteElement("DQ", quadrilateral, 0)
    Q1 = TensorProductElement(C, B)
    Q2 = TensorProductElement(D, A)
    hcurl_elements = [
        EnrichedElement(HCurl(AxB), HCurl(BxA)),
        EnrichedElement(HCurl(Q1), HCurl(Q2))
    ]
    for hcurl_element in hcurl_elements:
        assert hcurl_element in HCurl
        assert hcurl_element in L2
        assert hcurl_element in H0dx0dy
        assert hcurl_element not in H1
        assert hcurl_element not in H1dx1dy
        assert hcurl_element not in HDiv
        assert hcurl_element not in H2
        assert hcurl_element not in H2dx2dy


def test_varying_continuity_elements():
    P1DG_t = FiniteElement("DG", triangle, 1)
    P1DG_i = FiniteElement("DG", interval, 1)
    P1 = FiniteElement("CG", interval, 1)
    P2 = FiniteElement("CG", interval, 2)
    P3 = FiniteElement("CG", interval, 3)
    RT1 = FiniteElement("RT", triangle, 1)
    ARG = FiniteElement("ARG", triangle, 1)

    # Tensor product elements
    P1DGP2 = TensorProductElement(P1DG_t, P2)
    P1P1DG = TensorProductElement(P1, P1DG_i)
    P1DGP1 = TensorProductElement(P1DG_i, P1)
    RT1DG1 = TensorProductElement(RT1, P1DG_i)
    P2P3 = TensorProductElement(P2, P3)
    ARGP3 = TensorProductElement(ARG, P3)

    assert P1DGP2 in H1dz and P1DGP2 in L2
    assert P1DGP2 not in H1dh
    assert P1DGP1 in H1dy and P1DGP2 in L2
    assert P1P1DG in H1dx and P1P1DG in L2
    assert P1P1DG not in H1dx1dy
    assert RT1DG1 in H000 and RT1DG1 in L2
    assert P2P3 in H1dx1dy and P2P3 in H1
    assert ARG in H2dx2dy
    assert ARGP3 in H2dhH1dz
