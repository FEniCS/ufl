"""Tests for removing component tensors."""

from utils import LagrangeElement

from ufl import Coefficient, FunctionSpace, Mesh, as_vector, conditional, gt, indices, triangle
from ufl.algorithms.remove_component_tensors import IndexReplacer, remove_component_tensors
from ufl.classes import Index, Zero
from ufl.core.multiindex import FixedIndex


def test_index_replacer_handles_zero_without_free_indices():
    """Substituting a fixed index from a Zero leaves a scalar Zero."""
    index = Index()
    zero = Zero(shape=(), free_indices=(index.count(),), index_dimensions=(3,))

    result = IndexReplacer({index: FixedIndex(0)})(zero)

    assert result == Zero()


def test_remove_component_tensors_conditional_with_zero():
    """Fixing an index on a conditional with a Zero operand must not fail.

    Previously this raised ``ValueError: not enough values to unpack`` in
    ``IndexReplacer.zero``, because the Zero's only free index became fixed.
    """
    domain = Mesh(LagrangeElement(triangle, 1, (2,)))
    u = Coefficient(FunctionSpace(domain, LagrangeElement(triangle, 1, (2,))))
    (i,) = indices(1)
    # The conditional keeps its Zero operand, whose free index i becomes fixed
    expr = as_vector(conditional(gt(u[0], 0), 0 * u[i], u[i]), i)[0]
    assert remove_component_tensors(expr) == conditional(gt(u[0], 0), 0, u[0])
