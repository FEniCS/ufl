#!/usr/bin/env py.test
# -*- coding: utf-8 -*-

__authors__ = "David Ham"
__date__ = "2014-03-04"

import pytest
from ufl import FiniteElement, triangle
from ufl.sobolevspace import H2, H1, HDiv, HCurl, L2, SobolevSpace

# TODO: Add construction of all elements with periodic table notation here.


def test_inclusion():
    assert H2 < H1       # Inclusion
    assert not H2 > H1   # Not included
    assert HDiv <= HDiv  # Reflexivity
    assert H2 < L2       # Transitivity


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
    ]
    for l2_element in l2_elements:
        assert l2_element in L2
        assert l2_element not in H1
        assert l2_element not in HCurl
        assert l2_element not in HDiv
        assert l2_element not in H2


def test_contains_h1():
    h1_elements = [
        # Standard Lagrange elements:
        FiniteElement("CG", triangle, 1),
        FiniteElement("CG", triangle, 2),
        # Some special elements:
        FiniteElement("AW", triangle),
        FiniteElement("HER", triangle),
        FiniteElement("MTW", triangle),
    ]
    for h1_element in h1_elements:
        assert h1_element in H1
        assert h1_element in HDiv
        assert h1_element in HCurl
        assert h1_element in L2
        assert h1_element not in H2


def test_contains_h2():
    h2_elements = [
        FiniteElement("ARG", triangle, 1),
        FiniteElement("MOR", triangle),
    ]
    for h2_element in h2_elements:
        assert h2_element in H2
        assert h2_element in H1
        assert h2_element in HDiv
        assert h2_element in HCurl
        assert h2_element in L2


def test_contains_hdiv():
    hdiv_elements = [
        FiniteElement("RT", triangle, 1),
        FiniteElement("BDM", triangle, 1),
        FiniteElement("BDFM", triangle, 2),
    ]
    for hdiv_element in hdiv_elements:
        assert hdiv_element in HDiv
        assert hdiv_element in L2
        assert hdiv_element not in H1
        assert hdiv_element not in HCurl
        assert hdiv_element not in H2


def test_contains_hcurl():
    hcurl_elements = [
        FiniteElement("N1curl", triangle, 1),
        FiniteElement("N2curl", triangle, 1),
    ]
    for hcurl_element in hcurl_elements:
        assert hcurl_element in HCurl
        assert hcurl_element in L2
        assert hcurl_element not in H1
        assert hcurl_element not in HDiv
        assert hcurl_element not in H2
