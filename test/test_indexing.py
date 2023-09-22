import pytest

from ufl import Index, Mesh, SpatialCoordinate, outer, triangle
from ufl.classes import FixedIndex, Indexed, MultiIndex, Outer, Zero
from ufl.finiteelement import FiniteElement
from ufl.pull_back import identity_pull_back
from ufl.sobolevspace import H1


@pytest.fixture
def domain():
    return Mesh(FiniteElement("Lagrange", triangle, 1, (2, ), (2, ), identity_pull_back, H1))


@pytest.fixture
def x1(domain):
    x = SpatialCoordinate(domain)
    return x


@pytest.fixture
def x2(domain):
    x = SpatialCoordinate(domain)
    return outer(x, x)


@pytest.fixture
def x3(domain):
    x = SpatialCoordinate(domain)
    return outer(outer(x, x), x)


def test_annotated_literals():
    z = Zero(())
    assert z.ufl_shape == ()
    assert z.ufl_free_indices == ()
    assert z.ufl_index_dimensions == ()

    z = Zero((3,))
    assert z.ufl_shape == (3,)
    assert z.ufl_free_indices == ()
    assert z.ufl_index_dimensions == ()

    i = Index(count=2)
    j = Index(count=4)
    z = Zero((), (j, i), {i: 3, j: 5})
    assert z.ufl_shape == ()
    assert z.ufl_free_indices == (2, 4)
    assert z.ufl_index_dimensions == (3, 5)


def test_fixed_indexing_of_expression(x1, x2, x3):
    x0 = x1[0]
    x00 = x2[0, 0]
    x000 = x3[0, 0, 0]
    assert isinstance(x0, Indexed)
    assert isinstance(x00, Indexed)
    assert isinstance(x000, Indexed)
    assert isinstance(x0.ufl_operands[0], SpatialCoordinate)
    assert isinstance(x00.ufl_operands[0], Outer)
    assert isinstance(x000.ufl_operands[0], Outer)
    assert isinstance(x0.ufl_operands[1], MultiIndex)
    assert isinstance(x00.ufl_operands[1], MultiIndex)
    assert isinstance(x000.ufl_operands[1], MultiIndex)

    mi = x000.ufl_operands[1]
    assert len(mi) == 3
    assert mi.indices() == (FixedIndex(0),) * 3
