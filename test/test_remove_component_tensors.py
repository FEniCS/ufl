"""Tests for removing component tensors."""

from ufl.algorithms.remove_component_tensors import IndexReplacer
from ufl.classes import Index, Zero
from ufl.core.multiindex import FixedIndex


def test_index_replacer_handles_zero_without_free_indices():
    """Substituting a fixed index from a Zero leaves a scalar Zero."""
    index = Index()
    zero = Zero(shape=(), free_indices=(index.count(),), index_dimensions=(3,))

    result = IndexReplacer({index: FixedIndex(0)})(zero)

    assert result == Zero()
