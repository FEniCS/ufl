__authors__ = "David Ham"
__date__ = "2014-03-04"

from math import inf

from ufl import H1, H2, L2, HCurl, HDiv, HInf, triangle
from ufl.finiteelement import FiniteElement
from ufl.pullback import contravariant_piola, covariant_piola, identity_pullback
from ufl.sobolevspace import SobolevSpace  # noqa: F401
from ufl.sobolevspace import DirectionalSobolevSpace

# Construct directional Sobolev spaces, with varying smoothness in
# spatial coordinates
H0dx0dy = DirectionalSobolevSpace((0, 0))
H1dx1dy = DirectionalSobolevSpace((1, 1))
H2dx2dy = DirectionalSobolevSpace((2, 2))
Hinfdxinfdy = DirectionalSobolevSpace((inf, inf))
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
    assert H1 > H2
    assert L2 > H1


def test_directional_space_relations():
    assert H0dx0dy == L2
    assert H1dx1dy == H1
    assert H2dx2dy == H2
    assert H1dx1dy <= HDiv
    assert H1dx1dy <= HCurl
    assert H2dx2dy <= H1dx1dy
    assert H2dhH1dz < H1
    assert Hinfdxinfdy <= HInf
    assert Hinfdxinfdy < H2dx2dy
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
        FiniteElement("Discontinuous Lagrange", triangle, 0, (), identity_pullback, L2),
        FiniteElement("Discontinuous Lagrange", triangle, 1, (), identity_pullback, L2),
        FiniteElement("Discontinuous Lagrange", triangle, 2, (), identity_pullback, L2),
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
        FiniteElement("Lagrange", triangle, 1, (), identity_pullback, H1),
        FiniteElement("Lagrange", triangle, 2, (), identity_pullback, H1),
        # Some special elements:
        FiniteElement("MTW", triangle, 3, (2, ), contravariant_piola, H1),
        FiniteElement("Hermite", triangle, 3, (), "custom", H1),
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
        FiniteElement("ARG", triangle, 5, (), "custom", H2),
        FiniteElement("MOR", triangle, 2, (), "custom", H2),
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


def test_contains_hinf():
    hinf_elements = [
        FiniteElement("Real", triangle, 0, (), identity_pullback, HInf)
    ]
    for hinf_element in hinf_elements:
        assert hinf_element in HInf
        assert hinf_element in H2
        assert hinf_element in H2dx2dy
        assert hinf_element in H1
        assert hinf_element in H1dx1dy
        assert hinf_element in HDiv
        assert hinf_element in HCurl
        assert hinf_element in L2
        assert hinf_element in H0dx0dy


def test_contains_hdiv():
    hdiv_elements = [
        FiniteElement("Raviart-Thomas", triangle, 1, (2, ), contravariant_piola, HDiv),
        FiniteElement("BDM", triangle, 1, (2, ), contravariant_piola, HDiv),
        FiniteElement("BDFM", triangle, 2, (2, ), contravariant_piola, HDiv),
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
        FiniteElement("N1curl", triangle, 1, (2, ), covariant_piola, HCurl),
        FiniteElement("N2curl", triangle, 1, (2, ), covariant_piola, HCurl),
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
