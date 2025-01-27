"""Base class for dag visitors with UFL ``Expr`` type dispatch."""
from abc import ABC, abstractmethod
from functools import singledispatchmethod


class DAGVisitor(ABC):

    def __init__(self):
        self.cache = {}

    def __call__(self, node, *args):
        cache_key = (node, *args)
        try:
            return self.cache[cache_key]
        except KeyError:
            result = self.process(node, *args)
            self.cache[cache_key] = result
            return result

    @singledispatchmethod
    @abstractmethod
    def process(self, node, *args):
        """Process node by type."""

    def reuse_if_untouched(self, node, *args):
        """Reuse if touched."""
        new_ops = [self(child, *args) for child in node.ufl_operands]
        if all(nc == c for nc, c in zip(new_ops, node.ufl_operands)):
            return node
        else:
            return node._ufl_expr_reconstruct_(*new_ops)
