"""Base class for dag visitors with UFL Expr type dispatch."""

from abc import ABC, abstractmethod
from functools import singledispatchmethod

from ufl.classes import Expr


class DAGVisitor(ABC):
    """Base class for dag visitors with UFL-type dispatch."""

    def __init__(self) -> None:
        """Initialise."""
        self.cache = {}

    def __call__(self, node: Expr, *args) -> Expr:
        """Perform memoised DAG traversal with ``process`` singledispatch method.

        Args:
            node: `Expr` to start DAG traversal from.
            args: arguments to the ``process`` singledispatchmethod.

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
            args: arguments to the ``process`` singledispatchmethod.

        Returns:
            Processed `Expr`.

        """

    def reuse_if_untouched(self, node: Expr, *new_ufl_operands) -> Expr:
        """Reuse if touched.

        Args:
            node: `Expr` to start DAG traversal from.
            new_ufl_operands: new ufl_operands of ``node``.

        Returns:
            Processed `Expr`.

        """
        if all(nc == c for nc, c in zip(new_ufl_operands, node.ufl_operands)):
            return node
        else:
            return node._ufl_expr_reconstruct_(*new_ufl_operands)
