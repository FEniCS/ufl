"""Base class for dag visitors with UFL Expr type dispatch."""

from abc import ABC, abstractmethod
from functools import singledispatchmethod

from ufl.classes import Expr


class DAGVisitor(ABC):
    """Base class for dag visitors with UFL `Expr` type dispatch."""

    def __init__(self):
        """Initialise."""
        self.cache = {}

    def __call__(self, node: Expr, *args) -> Expr:
        """Perform memoised DAG traversal with ``process`` singledispatch method.

        Args:
            node: `Expr` to start DAG traversal from.
            args: `Sequence` of arguments to be passed to ``process``.

        Returns:
            Processed `Expr`.

        """
        cache_key = (node, *args)
        try:
            return self.cache[cache_key]
        except KeyError:
            result = self.process(node, *args)
            self.cache[cache_key] = result
            return result

    @singledispatchmethod
    @abstractmethod
    def process(self, node: Expr, *args) -> Expr:
        """Process node by type.

        Args:
            node: `Expr` to start DAG traversal from.
            args: `Sequence` of arguments to be passed to ``process``.

        Returns:
            Processed `Expr`.

        """

    def reuse_if_untouched(self, node: Expr, *args) -> Expr:
        """Reuse if touched.

        Args:
            node: `Expr` to start DAG traversal from.
            args: `Sequence` of arguments to be passed to ``process``.

        Returns:
            Processed `Expr`.

        """
        new_ops = [self(child, *args) for child in node.ufl_operands]
        if all(nc == c for nc, c in zip(new_ops, node.ufl_operands)):
            return node
        else:
            return node._ufl_expr_reconstruct_(*new_ops)
