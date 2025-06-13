"""Base class for dag traversers."""

from functools import singledispatchmethod, wraps
from typing import Union

from ufl.classes import Expr


class DAGTraverser:
    """Base class for DAG traversers.

    Args:
        compress: If True, ``result_cache`` will be used.
        visited_cache: cache of intermediate results; expr -> r = self.process(expr, ...).
        result_cache: cache of result objects for memory reuse, r -> r.

    """

    def __init__(
        self,
        compress: Union[bool, None] = True,
        visited_cache: Union[dict[tuple, Expr], None] = None,
        result_cache: Union[dict[Expr, Expr], None] = None,
    ) -> None:
        """Initialise."""
        self._compress = compress
        self._visited_cache = {} if visited_cache is None else visited_cache
        self._result_cache = {} if result_cache is None else result_cache

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
            return self._visited_cache[cache_key]
        except KeyError:
            result = self.process(node, *args)
            # Optionally check if r is in result_cache, a memory optimization
            # to be able to keep representation of result compact
            if self._compress:
                try:
                    # Cache hit: Use previously computed object, allowing current
                    # ``result`` to be garbage collected as soon as possible
                    result = self._result_cache[result]
                except KeyError:
                    # Cache miss: store in result_cache
                    self._result_cache[result] = result
            # Store result in cache
            self._visited_cache[cache_key] = result
            return result

    @singledispatchmethod
    def process(self, o: Expr, *args) -> Expr:
        """Process node by type.

        Args:
            o: `Expr` to start DAG traversal from.
            args: arguments to the ``process`` singledispatchmethod.

        Returns:
            Processed `Expr`.

        """
        raise AssertionError(f"Rule not set for {type(o)}")

    def reuse_if_untouched(self, o: Expr) -> Expr:
        """Reuse if touched.

        Args:
            o: `Expr` to start DAG traversal from.
            new_ufl_operands: new ufl_operands of ``o``.

        Returns:
            Processed `Expr`.

        """
        new_ufl_operands = [self(operand) for operand in o.ufl_operands]
        if all(nc == c for nc, c in zip(new_ufl_operands, o.ufl_operands)):
            return o
        else:
            return o._ufl_expr_reconstruct_(*new_ufl_operands)

    @staticmethod
    def postorder(method):
        """Postorder decorator.

        It is more natural for users to write a post-order singledispatchmethod
        whose arguments are ``(self, o, *processed_operands, *additional_args)``,
        while `DAGTraverser` expects one whose arguments are
        ``(self, o, *additional_args)``.
        This decorator takes the former and converts the latter, processing
        ``o.ufl_operands`` behind the users.

        """

        @wraps(method)
        def wrapper(self, o, *args):
            processed_operands = [self(operand) for operand in o.ufl_operands]
            return method(self, o, *processed_operands, *args)

        return wrapper

    @staticmethod
    def postorder_only_children(indices):
        """Postorder decorator with child indices.

        This decorator is the same as `DAGTraverser.postorder` except that the
        decorated method is only to take processed operands corresponding to ``indices``.

        """

        def postorder(method):
            @wraps(method)
            def wrapper(self, o, *args):
                processed_operands = [self(o.ufl_operands[i]) for i in indices]
                return method(self, o, *processed_operands, *args)

            return wrapper

        return postorder
